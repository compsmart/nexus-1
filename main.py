import argparse
import json
import os
import sys

# Suppress ALL progress bars and tokenizer warnings before any library import.
# Use direct assignment (not setdefault) so we always win even if a parent
# process already set a conflicting value.
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import logging
import queue
import threading
import time

from colorama import init, Fore, Style
from agent import AGIAgent

# Belt-and-suspenders tqdm suppression: sentence_transformers calls tqdm with
# an explicit disable=False kwarg which bypasses the TQDM_DISABLE env var.
# Monkey-patch AFTER all heavy imports so the class is loaded already.
try:
    import tqdm as _tqdm_mod
    _tqdm_orig_init = _tqdm_mod.tqdm.__init__
    def _tqdm_disabled_init(self, *args, **kwargs):
        kwargs["disable"] = True
        _tqdm_orig_init(self, *args, **kwargs)
    _tqdm_mod.tqdm.__init__ = _tqdm_disabled_init
    # Also patch tqdm.auto which sentence_transformers sometimes uses.
    try:
        import tqdm.auto as _tqdm_auto_mod
        _tqdm_auto_mod.tqdm.__init__ = _tqdm_disabled_init
    except Exception:
        pass
except Exception:
    pass

LISTEN_TIMEOUT_SECS = 8
LISTEN_PHRASE_LIMIT_SECS = 20

tts_engine = None
tts_enabled = False
tts_init_error = ""
try:
    import pyttsx3
    try:
        tts_engine = pyttsx3.init()
        tts_enabled = True
    except Exception as exc:  # pragma: no cover - runtime/audio backend dependent
        tts_init_error = str(exc)
except ImportError:
    tts_init_error = "pyttsx3 not installed"

sr = None
recognizer = None
stt_enabled = False
stt_init_error = ""
try:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    stt_enabled = True
except ImportError:
    stt_init_error = "SpeechRecognition not installed"
except Exception as exc:  # pragma: no cover - runtime dependency failure
    stt_init_error = str(exc)

init(autoreset=True)

# All log output goes to file only — never to the terminal so background
# threads (think-loop insights, encoder batches) never interrupt the prompt.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[logging.FileHandler(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexus.log"),
        encoding="utf-8",
    )],
)


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

class _Spinner:
    """A non-blocking terminal spinner written to stdout.
    Using stdout (not stderr) avoids console buffer interleaving on Windows
    where both streams share the same cursor position.
    """

    _FRAMES = r"|/-\\"

    def __init__(self, message: str = "Thinking"):
        self._message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "_Spinner":
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        # Overwrite the spinner line with spaces then move to a fresh new line.
        # Using \n (not \r) so subsequent stdout writes start on a clean line
        # and the spinner can never overwrite them.
        sys.stdout.write(f"\r{' ' * (len(self._message) + 8)}\r")
        sys.stdout.flush()

    def _run(self) -> None:
        idx = 0
        while not self._stop.is_set():
            frame = self._FRAMES[idx % len(self._FRAMES)]
            sys.stdout.write(f"\r  {self._message}  {frame}")
            sys.stdout.flush()
            idx += 1
            self._stop.wait(0.1)


class AsyncSpeaker:
    """Non-blocking TTS queue so speech does not block the input loop."""

    def __init__(self, enabled: bool):
        self._enabled = enabled
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        if self._enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="tts-loop")
            self._thread.start()

    def speak(self, text: str) -> None:
        if self._enabled and text:
            self._queue.put(text)

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop_event.set()
        self._queue.put("")
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            text = self._queue.get()
            if self._stop_event.is_set():
                break
            if not text:
                continue
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as exc:
                logging.warning("TTS error: %s", exc)

def _normalize_command(user_input: str) -> str:
    cmd = user_input.strip().lower()
    if cmd.startswith("/"):
        cmd = cmd[1:]
    return cmd

def _print_help() -> None:
    lines = [
        ("  /help   ", "Show available commands"),
        ("  /status ", "Show voice / speech status"),
        ("  /voice  ", "Toggle microphone input"),
        ("  /quit   ", "Exit and save memory"),
    ]
    print(Fore.YELLOW + Style.BRIGHT + "\n  Commands")
    print(Fore.YELLOW + "  " + "─" * 34)
    for cmd, desc in lines:
        print(Fore.CYAN + cmd + Fore.WHITE + desc)
    print()

def _print_status(use_voice: bool) -> None:
    def _yn(flag: bool) -> str:
        return (Fore.GREEN + "yes") if flag else (Fore.RED + "no")
    print(Fore.YELLOW + Style.BRIGHT + "\n  Status")
    print(Fore.YELLOW + "  " + "─" * 34)
    print(Fore.CYAN + "  Voice mode       " + Fore.WHITE + ("on" if use_voice else "off"))
    print(Fore.CYAN + "  Speech-to-text   " + _yn(stt_enabled))
    print(Fore.CYAN + "  Text-to-speech   " + _yn(tts_enabled))
    print()

def _toggle_voice(use_voice: bool) -> bool:
    if not stt_enabled:
        detail = f" ({stt_init_error})" if stt_init_error else ""
        print(
            Fore.RED
            + "Voice input unavailable. Install SpeechRecognition + PyAudio and verify microphone access."
            + detail
        )
        return False
    next_state = not use_voice
    print(Fore.YELLOW + f"Voice input {'enabled' if next_state else 'disabled'}.")
    return next_state

def listen() -> str:
    """Listens for speech input if enabled."""
    if stt_enabled and recognizer is not None:
        try:
            with sr.Microphone() as source:
                print(Fore.CYAN + "Listening...")
                audio = recognizer.listen(
                    source,
                    timeout=LISTEN_TIMEOUT_SECS,
                    phrase_time_limit=LISTEN_PHRASE_LIMIT_SECS,
                )
                try:
                    text = recognizer.recognize_google(audio)
                    print(Fore.GREEN + f"You said: {text}")
                    return text
                except sr.UnknownValueError:
                    print(Fore.RED + "Could not understand audio.")
                except sr.RequestError as e:
                    print(Fore.RED + f"Could not request results; {e}")
        except sr.WaitTimeoutError:
            print(Fore.YELLOW + "No speech detected. Try again.")
        except OSError as exc:
            print(Fore.RED + f"Microphone unavailable: {exc}")
            logging.warning("Microphone error: %s", exc)
        except Exception as exc:
            print(Fore.RED + f"Voice input error: {exc}")
            logging.warning("Voice input error: %s", exc)
    return ""

def main() -> int:
    # --- Parse command-line arguments (UI mode support) ---------------
    parser = argparse.ArgumentParser(description="Nexus AGI Research Agent")
    parser.add_argument("--message", type=str, help="Non-interactive mode: send a single message and exit")
    args = parser.parse_args()

    # In non-interactive mode redirect stdout to stderr so all the banner,
    # spinner and warning text doesn't pollute the JSON response the UI parses.
    _real_stdout = sys.stdout
    if args.message:
        sys.stdout = sys.stderr

    # --- Banner -------------------------------------------------------
    width = 44
    border = Fore.MAGENTA + Style.BRIGHT + "  " + "═" * width
    print()
    print(border)
    print(Fore.MAGENTA + Style.BRIGHT + "  ║" + Fore.WHITE + "  NEXUS  " + Fore.CYAN + "│" + Fore.WHITE + "  AGI Research Agent" + " " * 13 + Fore.MAGENTA + Style.BRIGHT + "║")
    print(border)
    print()

    # --- Capability warnings (compact, one-line each) -----------------
    if not stt_enabled:
        detail = f" ({stt_init_error})" if stt_init_error else ""
        print(Fore.YELLOW + f"  [STT unavailable{detail}]")
    if not tts_enabled:
        detail = f" ({tts_init_error})" if tts_init_error else ""
        print(Fore.YELLOW + f"  [TTS unavailable{detail}]")

    # --- Init agent ---------------------------------------------------
    spinner = _Spinner("Initializing")
    spinner.start()
    try:
        agent = AGIAgent()
    except Exception as exc:
        spinner.stop()
        logging.exception("Failed to initialize AGI agent")
        print(Fore.RED + f"\n  Error: failed to initialize agent: {exc}")
        return 1
    spinner.stop()

    speaker = AsyncSpeaker(tts_enabled)
    agent.start()

    # --- Non-interactive mode (for UI) --------------------------------
    if args.message:
        try:
            response = agent.interact(args.message)
        except Exception as e:
            logging.exception("Error processing message")
            response = None
            sys.stdout = _real_stdout
            print(json.dumps({"response": "", "error": str(e)}))
            agent.stop()
            return 1
        finally:
            agent.stop()
        # Restore real stdout and emit clean JSON — nothing else goes here.
        sys.stdout = _real_stdout
        print(json.dumps({"response": response}))
        return 0

    # --- Interactive mode (normal CLI) --------------------------------
    print(Fore.GREEN + Style.BRIGHT + "  Nexus online." + Style.RESET_ALL
          + Fore.WHITE + "  Type " + Fore.CYAN + "/help" + Fore.WHITE + " for commands.\n")

    use_voice = False

    try:
        while True:
            # --- Input ------------------------------------------------
            if use_voice and stt_enabled:
                user_input = listen()
                if not user_input:
                    continue
            else:
                try:
                    sys.stdout.write(Fore.CYAN + Style.BRIGHT + "You  " + Fore.WHITE + Style.NORMAL + ": ")
                    sys.stdout.flush()
                    user_input = input()
                except EOFError:
                    print(Fore.RED + "\n  Input stream closed.")
                    break

            # --- Commands ---------------------------------------------
            command = _normalize_command(user_input)
            if command in ("quit", "exit"):
                break
            if command in ("help", "h", "commands"):
                _print_help()
                continue
            if command in ("voice", "mic"):
                use_voice = _toggle_voice(use_voice)
                continue
            if command == "status":
                _print_status(use_voice)
                continue
            if not user_input.strip():
                continue

            # --- Stream response with spinner -------------------------
            spinner = _Spinner("Thinking")
            spinner.start()

            first_chunk = True
            full_response = ""
            try:
                for chunk in agent.stream_interact(user_input):
                    if first_chunk:
                        spinner.stop()
                        # Response header: spinner.stop() leaves cursor at col 0
                        # of the now-blank spinner line, so write directly here.
                        sys.stdout.write(
                            Fore.GREEN + Style.BRIGHT + "Nexus" + Fore.WHITE
                            + Style.NORMAL + ": " + Style.RESET_ALL
                        )
                        sys.stdout.flush()
                        first_chunk = False
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    full_response += chunk
            except Exception as exc:
                spinner.stop()
                logging.exception("Turn failed")
                print(Fore.RED + f"\n  Error: {exc}")
                continue

            if first_chunk:
                # Nothing streamed (empty response or all-tool-call turn).
                spinner.stop()

            # Newline after response + subtle separator.
            print("\n" + Fore.MAGENTA + "  " + "─" * 42)

            speaker.speak(full_response)

    except KeyboardInterrupt:
        print(Fore.RED + "\n\n  Interrupted.")
    finally:
        speaker.stop()
        agent.stop()
        print(Fore.MAGENTA + Style.BRIGHT + "\n  Nexus offline." + Style.RESET_ALL)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

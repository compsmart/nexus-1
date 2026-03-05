"""Composite case set: recall + multihop + reasoning + learning questions."""

from __future__ import annotations

from typing import Dict, List

_CASES = [
    # Memory recall
    {"id": "C001", "q": "What colour does Alice prefer?", "keywords": ["blue"], "category": "memory",
     "context_docs": ["Alice prefers blue.", "Bob prefers red.", "Charlie prefers green."]},
    {"id": "C002", "q": "Where does Bob work?", "keywords": ["acme"], "category": "memory",
     "context_docs": ["Bob works at Acme.", "Alice works at Globex.", "Diana works at Initech."]},
    {"id": "C003", "q": "What pet does Charlie own?", "keywords": ["parrot"], "category": "memory",
     "context_docs": ["Charlie owns a parrot.", "Diana owns a cat.", "Edward owns a dog."]},
    {"id": "C004", "q": "What city does Diana live in?", "keywords": ["tokyo"], "category": "memory",
     "context_docs": ["Diana lives in Tokyo.", "Edward lives in Berlin.", "Fiona lives in Paris."]},
    {"id": "C005", "q": "What instrument does Edward play?", "keywords": ["violin"], "category": "memory",
     "context_docs": ["Edward plays the violin.", "Fiona plays the piano.", "George plays guitar."]},
    {"id": "C006", "q": "What is Fiona's favourite food?", "keywords": ["sushi"], "category": "memory",
     "context_docs": ["Fiona loves sushi.", "George loves pasta.", "Hannah loves tacos."]},

    # Multi-hop
    {"id": "C007", "q": "Alice knows Bob, Bob knows Charlie. Starting from Alice, following 2 links, who do you reach?",
     "keywords": ["charlie"], "category": "multi_hop",
     "context_docs": ["Alice KNOWS Bob.", "Bob KNOWS Charlie.", "Charlie KNOWS Diana."]},
    {"id": "C008", "q": "X trusts Y, Y trusts Z, Z trusts W. If X=Diana, Y=Edward, Z=Fiona, W=George, who does Diana reach in 3 steps?",
     "keywords": ["george"], "category": "multi_hop",
     "context_docs": ["Diana TRUSTS Edward.", "Edward TRUSTS Fiona.", "Fiona TRUSTS George."]},
    {"id": "C009", "q": "Alpha links to Bravo, Bravo links to Charlie. Following 2 links from Alpha, who do you reach?",
     "keywords": ["charlie"], "category": "multi_hop",
     "context_docs": ["Alpha LINKS Bravo.", "Bravo LINKS Charlie.", "Delta LINKS Echo."]},
    {"id": "C010", "q": "Person1 KNOWS Person2, Person2 KNOWS Person3, Person3 KNOWS Person4. Following 3 links from Person1?",
     "keywords": ["person4"], "category": "multi_hop",
     "context_docs": ["Person1 KNOWS Person2.", "Person2 KNOWS Person3.", "Person3 KNOWS Person4."]},
    {"id": "C011", "q": "Sam befriends Tom, Tom befriends Uma. Following 2 BEFRIENDS links from Sam?",
     "keywords": ["uma"], "category": "multi_hop",
     "context_docs": ["Sam BEFRIENDS Tom.", "Tom BEFRIENDS Uma.", "Uma BEFRIENDS Vera."]},
    {"id": "C012", "q": "If A->B->C->D->E, following 4 links from A, who do you reach?",
     "keywords": ["e"], "category": "multi_hop",
     "context_docs": ["A LINKS B.", "B LINKS C.", "C LINKS D.", "D LINKS E."]},

    # Reasoning
    {"id": "C013", "q": "If all birds can fly and a penguin is a bird, can a penguin fly according to this rule?",
     "keywords": ["yes"], "category": "reasoning",
     "context_docs": ["Rule: All birds can fly."]},
    {"id": "C014", "q": "X is taller than Y, Y is taller than Z. Is X taller than Z?",
     "keywords": ["yes"], "category": "reasoning",
     "context_docs": ["X is taller than Y.", "Y is taller than Z."]},
    {"id": "C015", "q": "If A implies B, and B implies C, does A imply C?",
     "keywords": ["yes"], "category": "reasoning",
     "context_docs": ["If A then B.", "If B then C."]},
    {"id": "C016", "q": "Three boxes: red, blue, green. The key is not in the red box and not in the green box. Where is the key?",
     "keywords": ["blue"], "category": "reasoning",
     "context_docs": ["There are three boxes: red, blue, green.", "The key is not in the red box.", "The key is not in the green box."]},
    {"id": "C017", "q": "If it rains, the ground is wet. The ground is wet. Did it necessarily rain?",
     "keywords": ["no"], "category": "reasoning",
     "context_docs": ["If it rains, the ground is wet."]},
    {"id": "C018", "q": "A farmer has 17 sheep. All but 9 run away. How many are left?",
     "keywords": ["9"], "category": "reasoning",
     "context_docs": []},

    # Learning
    {"id": "C019", "q": "Using Rule A, what is CODE(dog)?", "keywords": ["EPH"], "category": "learning",
     "context_docs": ["Rule A: CODE(x) shifts each letter forward by one and uppercases output. Example: CODE(cat)=DBU."]},
    {"id": "C020", "q": "Which city is the owner of Project Kappa located in?", "keywords": ["seoul"], "category": "learning",
     "context_docs": ["Project Kappa is owned by Lina.", "Lina is located in Seoul."]},
    {"id": "C021", "q": "What is the emergency protocol token?", "keywords": ["channel-7"], "category": "learning",
     "context_docs": ["Emergency protocol token is channel-7."]},
    {"id": "C022", "q": "Using Rule A, what is CODE(hi)?", "keywords": ["IJ"], "category": "learning",
     "context_docs": ["Rule A: CODE(x) shifts each letter forward by one and uppercases output. Example: CODE(cat)=DBU."]},
    {"id": "C023", "q": "Who owns Project Kappa?", "keywords": ["lina"], "category": "learning",
     "context_docs": ["Project Kappa is owned by Lina.", "Lina is located in Seoul."]},
    {"id": "C024", "q": "What protocol should be used in an emergency?", "keywords": ["channel-7"], "category": "learning",
     "context_docs": ["Emergency protocol token is channel-7."]},
]


def composite_cases(num_cases: int = 24) -> List[Dict]:
    """Return up to *num_cases* composite benchmark cases."""
    return _CASES[:num_cases]

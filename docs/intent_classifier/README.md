# Intent Classifier Agent

**Source file:** [`src/minecraft_ai_helper/agents/intent_classifier.py`](../../src/minecraft_ai_helper/agents/intent_classifier.py)

## Purpose

The intent classifier is the **first step of every query**. It runs before any other agent and answers two questions:

1. What kind of question is this? (the *intent*)
2. Which agents should answer it, and with what search query?

It keeps the rest of the pipeline focused — a crafting question doesn't need a Tavily web search, and a "best strategy" question doesn't need official technical docs.

## Input / Output

**Input:** raw player question (string)

**Output:** `IntentResult`
```python
{
  "intent": "crafting",
  "agents_to_invoke": ["wiki_rag", "official_docs"],
  "search_query": "diamond pickaxe crafting recipe ingredients"
}
```

## Intent Categories

| Intent | Covers |
|---|---|
| `crafting` | Recipes, crafting tables, ingredients, anvil upgrades |
| `combat` | Fighting mobs, weapons, armour, PvP, boss strategies |
| `redstone` | Circuits, contraptions, pistons, observers, hoppers |
| `biome` | Biome locations, structures, generation, climate |
| `enchanting` | Enchantment tables, books, anvils, enchant levels |
| `farming` | Crop farms, animal breeding, mob farms, XP farms |
| `building` | Building tips, material choices, structural blocks |
| `mechanic` | Game systems: hunger, sleep, respawn, weather, XP |
| `lore` | Story, history, Easter eggs, developer notes |
| `general` | Anything that doesn't clearly fit above |

## Agent Routing Rules

| Agent | When invoked |
|---|---|
| `wiki_rag` | Always — it's the primary factual source |
| `official_docs` | For `crafting`, `enchanting`, `mechanic`, `redstone` (technical accuracy matters) |
| `community` | For `combat`, `farming`, `building`, or questions containing "best", "strategy", "meta", "efficient", "tips" |

## How It Works

Sends the player's question to the LLM with a structured system prompt that defines the categories and routing rules. The model is instructed to respond with **only a JSON object** — no prose — to keep latency low.

Uses `use_thinking=False` intentionally: classification is a simple structured task that doesn't benefit from extended reasoning, and keeping it fast matters since it blocks all downstream agents.

Falls back to `intent=general, agents=["wiki_rag"]` if the model returns unparseable JSON.

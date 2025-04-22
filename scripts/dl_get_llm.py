from openai import OpenAI
import os
from dotenv import load_dotenv
import streamlit as st
import wikipediaapi

load_dotenv()


def get_wiki_info(artist):
    context = ''
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='MyProjectName (merlin@example.com)', language='en')
    page_py = wiki_wiki.page(artist)
    for section in page_py.sections:
        if section.title.lower() in ["biography", "early life", "career", "art", "painting"]:
            context += (section.text)

    return context

def get_art_explanation(results, client):

    artist_context = get_wiki_info(results['metadata']['artist'])
    prompt = f"""
You are an expert art historian and skilled writer.

Given the painting metadata below, generate a **rich, structured, and beautifully formatted Markdown** explanation with:
- 🎨 Title and artist as a heading
- 🖼️ A short paragraph on what it represents
- 🕰️ When and why it was painted (if known)
- 🌍 Cultural or historical context

**Return valid Markdown only.**

### Painting Metadata:
- Title: {results['metadata']['title']}
- Artist: {results['metadata']['artist']}
- Style: {results['metadata']['style']}
- Genre: {results['metadata']['genre']}

### Artist Background:
{artist_context[:2500]}

"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content
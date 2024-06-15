import getpass
import os
import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
import random

# Environment setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
if "LANGCHAIN_API_KEY" not in os.environ:
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("langchain api key here")

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("open ai api key here")

llm = ChatOpenAI(
    model = "gpt-3.5-turbo-0125",
    temperature = 0.1,
    max_tokens = 512
)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE_old = """
You are playing a game of Wordle.

For each guess, I will provide feedback as indicated below:
(G) = Correct letters in correct position
(Y) = Correct letters in incorrect position
(_) = Incorrect letter

Here are the rules:
Guess one word at a time.
Only use words with 5 letters.
Only use small capital letters.
Guess the word in six attempts or less.
Answers are never plurals.
Each guess must be a valid five-letter word.
Letters may be used more than once.

Tips: 
Avoid guessing the same word.
Keep (G)
Move around (Y)
Avoid (_)
Learn the possible words here: {context}

Your objective is to win the game by getting the feedback "G G G G G". Follow the rules and the tips.
Guess: {guess}
Feedback: {feedback}

Guess the word:
"""

PROMPT_TEMPLATE = """
You are playing a game of Wordle. I will teach you how to play Wordle.

Below are the rules of the game:
1. Guess one word at a time.
2. Only use words with 5 letters.
3. Only use small capital letters.
4. Guess the word in six attempts or less.
5. Answer are never plurals.
6. Each guess must be a valid five-letter word.
7. Letters may be used more than once.

Here is how to win:
1. Guess the word. Follow the rules.
2. If you get a "G" in feedback, keep the letter.
3. If you get a "Y" in feedback, place the letter somewhere else.
4. If you get a "_" in feedback, do not use the letter again.
5. If you get a "G G G G G" in feedback, you win!

Your first guess: {guess}
Feedback: {feedback}

Guess the word:
"""

def load_documents(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file]
    return words


def play_wordle(db, word):
    print(f"Playing Wordle for the word: {word}")
    feedback = ""
    guess = "salet" # or salet
    attempts = 0

    # Mask the word in the prompt template
    masked_word = '*' * len(word)

    while guess.lower() != word.lower() and attempts < 6:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        # Pass the masked word to the prompt template
        prompt = prompt_template.format(target_word=masked_word, guess=guess, feedback=feedback)
        print(prompt)
        response = llm.invoke(prompt)
        print(f"LLM's Response: {response.content}")

        # Extract the guess from the response content
        guess_lines = response.content.strip().split("\n")
        guess = guess_lines[0].strip().replace('Guess: ', '')
        print(f"LLM's Guess: {guess}")

        # Remove any extra messages from the guess
        if "Congratulations" in guess:
            guess = word

        # Provide feedback based on the presence of letters in the word
        feedback = provide_feedback_presence(word, guess)
        print(f"Feedback: {feedback}")

        attempts += 1

    if guess.lower() == word.lower():
        return True
    else:
        return False

def provide_feedback_presence(target_word, guess):
    green = 0
    yellow = 0
    gray = 0
    feedback = []
    min_length = min(len(target_word), len(guess))  # Get the minimum length of the target word and guess
    for i in range(min_length):
        if guess[i] == target_word[i]:
            feedback.append("G")  # Letter is in the correct position
            green +=1
        elif guess[i] in target_word and guess[i] != target_word[i]:
            feedback.append("Y")  # Letter is present in the word but not in the correct position
            yellow +=1
        else:
            feedback.append("_")    # Letter is not present in the word
            gray +=1
    feedback.append(f"(G) = {green} letters, (Y) = {yellow} letters, (_) = {gray} letters")
    return " ".join(feedback)


def main():
    # Load the target words from the text file
    file_path = r"previous_words.txt"
    words = load_documents(file_path)

    # Prepare the DB
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    successful_games = 0
    games = 0
    attempts = 0

    while games < 6:  # Limiting the total number of games to 6
        word = random.choice(words)  # Select a random word from the list
        print(f"Playing Wordle for the word: {word}")
        result = play_wordle(db, word)

        if result:
            print(f"Successfully guessed the word '{word}' in {attempts} attempts.")
            successful_games += 1
            games +=1
        else:
            print(f"Failed to guess the word '{word}' in 6 attempts. Moving to the next word.")
            games +=1

    print(f"Game finished: {successful_games} successful guesses out of {games} games.")  # Corrected the print statement


if __name__ == "__main__":
    main()
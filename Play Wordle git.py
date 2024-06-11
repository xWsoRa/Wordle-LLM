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
    model = "gpt-4o",
    temperature = 0.1,
    max_tokens = 1024
)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are playing a game of Wordle. The target word is {target_word}.
Try to guess the word in as few attempts as possible. After each guess, I will provide feedback indicating how many letters are correct and in the correct position (G), how many letters are correct but in the wrong position (Y), and how many letters are incorrect (_). Keep guessing until you find the word.
Instructions for Wordle:
Only guess small caps letters
You have to guess the Wordle in six goes or less
A correct letter in the correct position provides the feedback G
A correct letter in the wrong position provides the feedback Y
An incorrect letter or letter that is not in the word provides the feedback _
Letters can be used more than once but you cannot use the same word more than once per game
Answers are never plurals
Each guess must be a valid five-letter word.
You must only use 5 letter words do not explain your reasoning

Tips: use words that have different letters to see if they are in the word

Example of a game:
I: Play Wordle 5 times.
O: Stair.
I: 'S', 't', 'i', and 'r' are _, and 'a' is Y.
O: Beast.
I: 'B', 's', and 't' are _, and 'e' and 'a' are Y.
O: Eager.
I: 'r' and the second 'e' are _, and 'e', 'a', and 'g' are G.
O: Eagle.
I: Correct

Context: {context}

Guess: {guess}

Feedback: {feedback}

Make your next guess:
"""

def load_documents(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file]
    return words


def play_wordle(db, word):
    print(f"Playing Wordle for the word: {word}")
    feedback = ""
    guess = "crane"  # Set the initial guess to "salet"
    attempts = 0

    # Mask the word in the prompt template
    masked_word = '*' * len(word)

    # Search the DB for relevant context
    db_results = db.similarity_search_with_relevance_scores(word, k=3)
    if len(db_results) == 0 or db_results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return False

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in db_results])

    while guess.lower() != word.lower() and attempts < 6:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        # Pass the masked word to the prompt template
        prompt = prompt_template.format(target_word=masked_word, context=context_text, guess=guess, feedback=feedback)
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
    feedback.append(f"Congratulations {green} letters in the correct position, almost correct with {yellow} letters in the word but wrong position, completely wrong with {gray} letters.")
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
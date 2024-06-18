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
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangChain API key here")

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key here")

llm = ChatOpenAI(
    model = "gpt-4o", # MODIFIABLE - other models: gpt-3.5-turbo-0125, gpt-4o
    temperature = 1.3, # 1.3 = best temp 
    max_tokens = 1024,
    top_p = 0.1 # 0.1 = best top_p
)

### MODIFIABLE BELOW THIS LINE ###
GAME_RULES = """
Guess one word at a time.
Do not repeat words.
Only use words with 5 letters.
Guess the word in six attempts or less.
Answer are never plurals.
Each guess must be a valid five-letter word.
Letters may be used more than once.
"""

WIN_CONDITIONS = """
Here is how to win:
Guess the word. Follow the rules.
Put the correct letters in the correct positions.
If all of the letters are correct you win
"""


PROMPT_TEMPLATE = f"""
You are the best Wordle player, guess the 5 letter word

Below are the rules of the game:
{GAME_RULES}

Conditions to win:
{WIN_CONDITIONS}


Previous guesses and feedback:
{{history}}

Guess a different 5 letter target word only do not write anything else:
"""
### DO NOT MODIFY FURTHER BELOW ###

def load_documents(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file]
    return words


def play_wordle(word):
    print(f"Playing Wordle for the word: {word}")
    feedback = ""
    guess = ""
    attempts = 0
    history = []

    # Mask the word in the prompt template
    masked_word = '*' * len(word)

    # Open file to append history
    with open(r"wordle_history_prompt_3.txt", 'a') as history_file:
        while guess.lower() != word.lower() and attempts < 10:
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            
            # Format history for the prompt template
            formatted_history = "\n".join(history)
            
            # Pass the masked word and history to the prompt template
            prompt = prompt_template.format(target_word=masked_word, guess=guess, feedback=feedback, history=formatted_history)
            print(prompt)
            response = llm.invoke(prompt)
            print(f"LLM's Response: {response.content}")

            # Extract the guess from the response content
            guess_lines = response.content.strip().split("\n")
            guess = guess_lines[0].strip().replace('Guess: ', '')
            print(f"LLM's Guess: {guess}")

            # Provide feedback based on the presence of letters in the word
            feedback = provide_feedback_presence(word, guess)
            print(f"Feedback: {feedback}")

            # Update history
            history.append(f"Guess: {guess} - Feedback: {feedback}")

            # Write history to file
            history_file.write(f"Word: {word}, Attempt: {attempts + 1}, Guess: {guess}, Feedback: {feedback}\n")

            attempts += 1

    if guess.lower() == word.lower():
        return True, attempts
    else:
        return False, attempts


def provide_feedback_presence(target_word, guess):
    feedback = []
    guess = guess.lower()
    target_word = target_word.lower()
    min_length = min(len(target_word), len(guess))  # Get the minimum length of the target word and guess

    for i in range(min_length):
        if guess[i] == target_word[i]:
            feedback.append(f"The letter {guess[i]} is in the correct position")  # only text is MODIFIABLE
        elif guess[i] in target_word and guess[i] != target_word[i]:
            feedback.append(f"The letter {guess[i]} is in the target word but in wrong position")  # only text is MODIFIABLE
        else:
            feedback.append(f"The letter {guess[i]} is not in the target word")  # only text is MODIFIABLE

    feedback_str = ", ".join(feedback)
    return feedback_str

def main():
    # Load the target words from the text file
    file_path = r"previous_words.txt"
    words = load_documents(file_path)


    successful_games = 0
    games = 0

    while games < 5:  # Alter the total number of games as needed. 10 for testing prompt designs, 100 for final quantitative result.
        word = random.choice(words)  # Select a random word from the list
        print(f"Playing Wordle for the word: {word}")
        result, attempts = play_wordle(word)

        if result:
            print(f"Successfully guessed the word '{word}' in {attempts} attempts.")
            successful_games += 1
        else:
            print(f"Failed to guess the word '{word}' in 6 attempts.")

        games += 1

    print(f"Game finished: {successful_games} successful guesses out of {games} games.")


if __name__ == "__main__":
    main()
    
def is_unique(word_list):
  """
  Checks if all words in a dictionary are unique.

  Args:
    word_list: A dictionary where keys are words and values are their counts.

  Returns:
    True if all words are unique, False otherwise.
  """

  # Check if any word count is greater than 1.
  print("Inside thefunction")
  for count in word_list.values():
    if count > 1:
      return False
  return True

# Example usage
word_dict = {"apple": 1, "banana": 2, "orange": 1}
print(is_unique(word_dict))  # Output: False

word_dict = {"apple": 1, "banana": 1, "orange": 1}
print(is_unique(word_dict))  # Output: True

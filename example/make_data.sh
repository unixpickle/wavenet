#!/bin/bash
#
# Requires a Mac with the `say` command.

SAY_CMD=say
DICT_FILE=/usr/share/dict/words
DATA_DIR=data

if ! type "$SAY_CMD" 2>/dev/null >/dev/null; then
  echo "Say command not found: `$SAY_CMD`" >&2
  exit 1
fi

if ! [ -f "$DICT_FILE" ]; then
  echo "Dictionary file not found: $DICT_FILE" >&2
  exit 1
fi

if [ -d "$DATA_DIR" ]; then
  echo "Data directory not found: $DATA_DIR" >&2
  exit 1
fi

mkdir "$DATA_DIR"

for line in $(cat $DICT_FILE); do
  echo $RANDOM $RANDOM $line
done | sort -n | cut -f 2- -d ' ' | sort -n | cut -f 2- -d ' ' >"$DATA_DIR"/words.txt

while read word1; do
  read word2
  $SAY_CMD -o "$DATA_DIR/${word1}_$word2.wav" --data-format=LEI16@44100 "$word1 $word2"
done < "$DATA_DIR/words.txt"

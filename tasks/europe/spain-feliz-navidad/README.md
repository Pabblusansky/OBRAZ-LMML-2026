# ¡Feliz Navidad!

| Field | Value |
|---|---|
| Country | Spain |
| Difficulty | Easy |
| Category | data validation — rule-based filtering |
| Solved by | Artem |

## Task
21 pilots brought back 6 kinds of souvenirs (`files/office_price_guide.csv`, receipts in
`files/purchases.csv`). A souvenir counts only if its unit price is within the fair range
(inclusive). On the office board each pilot has a column, and HR marks the fair purchases.
Together the marks spell a word.

## Approach (`main.py`)
Unit price = total paid ÷ quantity. A receipt counts if the price is within
[`Fair_min`, `Fair_max`] **and** the item was bought in its home city. Then plot the board
(column = pilot, row = item on the list) and read the word off `board.png`.

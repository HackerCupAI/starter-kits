If Alice reaches row \(R\) before Bob reaches row \(C\), then it's game over for Alice. Since each player now wants to get to the finish as slowly as possible, both have a simple dominating strategy of only moving \(1\) unit in their direction each turn, and \(R\) and \(C\) are the only things that matter.

If \(R \le C\), Bob can always force Alice to reach row \(R\) first by moving \(1\) unit right at a time. Alice also only moves \(1\) unit at a time, because if she moves any faster, she'll just get stuck sooner. 

Example: Alice moves first:
```
[ ][ ][ ][ ]
[x][ ][ ][ ]
[ ][ ][ ][ ]
```

Bob:
```
[ ][ ][ ][ ]
[ ][x][ ][ ]
[ ][ ][ ][ ]
```

Alice moves, and gets stuck to watch Bob stroll to the finish line:
```
[ ][ ][ ][ ]
[ ][ ][ ][ ]
[ ][x][ ][ ]
```

Conversely, if \(R > C\), then Alice can always force a win by moving \(1\) step at a time. Therefore we output "`YES`" if and only if \(R > C\), regardless of the values of \(A\) and \(B\).

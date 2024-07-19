*Nim Sum Dim Sum*, a bustling local dumpling restaurant, has two game theory-loving servers named, you guessed it, Alice and Bob. Its dining area can be represented as a two-dimensional grid of \(R\) rows (numbered \(1..R\) from top to bottom) by \(C\) columns (numbered \(1..C\) from left to right\).

Currently, both of them are standing at coordinates \((1, 1)\) where there is a big cart of dim sum. Their job is to work together to push the cart to a customer at coordinates \((R, C)\). To make the job more interesting, they've turned it into a game.

Alice and Bob will take turns pushing the cart. On Alice's turn, the cart must be moved between \(1\) and \(A\) units down. On Bob's turn, the cart must be moved between \(1\) and \(B\) units to the right. The cart may not be moved out of the grid. If the cart is already at row \(R\) on Alice's turn or column \(C\) on Bob's turn, then that person loses their turn.

The "winner" is the person to ultimately move the cart to \((R, C)\) and thus get all the recognition from the customer. Alice pushes first. Does she have a guaranteed winning strategy?


# Constraints

\(1 \leq T \leq 500\)
\(2 \leq R, C \leq 10^9\)
\(1 \leq A < R\)
\(1 \leq B < C\)


# Input Format

Input begins with an integer \(T\), the number of test cases. Each case will contain one line with four space-separated integers, \(R\), \(C\), \(A\), and \(B\).


# Output Format

For the \(i\)th test case, print `"Case #i: "` followed by `"YES"` if Alice has a guaranteed winning strategy, or `"NO"` otherwise.


# Sample Explanation

The first case is depicted below, with Alice's moves in red and Bob's in blue. Alice moves down, and Bob moves right to win immediately. There is no other valid sequence of moves, so Alice has no guaranteed winning strategy.

{{PHOTO_ID:842253013944047|WIDTH:500}}

The second case is depicted below. One possible guaranteed winning strategy is if Alice moves \(3\) units down, then Bob can only move \(1\) unit, and finally Alice can win with \(1\) unit.

{{PHOTO_ID:852013469652032|WIDTH:500}}

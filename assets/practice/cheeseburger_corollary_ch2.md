*This problem shares some similarities with A1, with key differences in bold.*

Problem solving skills are applicable to many daily musings. For instance, you might ponder over shared birthdays, bird houses, mapmaking, or ordering an exact number of chicken nuggets. Naturally, another great question to ponder is: how many deckers of a cheeseburger you could build if you spent your entire salary on fast food!

Specifically, you're interested in building a \(K\)-decker cheeseburger, which alternates between buns, cheese, and patty starting and ending with a bun. **Buying a single cheeseburger costs \(A\) dollars and a double cheeseburger costs \(B\) dollars**. Each provides you with two buns, though a single provides one patty and one cheese, while a double provides two patties and two cheese.

{{PHOTO_ID:181863494933248|WIDTH:600}}

You'd like to know **the biggest \(K\) for which you can build a \(K\)-decker cheeseburger by spending at most \(C\) dollars**.


# Constraints

\(1 \leq T \leq 65\)
\(1 \leq A, B, C \leq 10^{16}\)


# Input Format

Input begins with an integer \(T\), the number of test cases. Each case contains one line with three space-separated integers, \(A\), \(B\) and \(C\).


# Output Format

For the \(i\)th test case, print "`Case #i:` " followed by the largest possible \(K\) for which you can build a \(K\)-decker cheeseburger, or \(0\) if you cannot build even a \(1\)-decker cheeseburger.


# Sample Explanation

In the first case, you can buy \(1\) single and \(1\) double cheeseburger. This gives you \(4\) buns, \(3\) slices of cheese, and \(3\) patties, exactly enough for a \(3\)-decker cheeseburger.

In the second case, you can only afford to build a \(1\)-decker cheeseburger. That's coincidentally identical to a single cheeseburger.

In the third case, you only have \(\$1\), while a single costs \(\$2\) and a double costs \(\$3\). No fast food for you today.

In the fourth case, our best possible strategy is to spend all our money on \(100\) doubles, which we can use to build a \(199\)-decker cheeseburger.

In the fifth case, our best possible strategy is to spend all our money on \(100\) singles, which we can use to build a \(100\)-decker cheeseburger.

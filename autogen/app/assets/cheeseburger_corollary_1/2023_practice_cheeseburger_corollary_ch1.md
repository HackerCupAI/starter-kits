*This problem shares some similarities with A2, with key differences in bold.*

Problem solving skills are applicable to many daily musings. For instance, you might ponder over shared birthdays, bird houses, mapmaking, or ordering an exact number of chicken nuggets. Naturally, another great question to ponder is: how many deckers of a cheeseburger you could build if you spent your entire salary on fast food!

Specifically, you're interested in building a \(K\)-decker cheeseburger, which alternates between buns, cheese, and patty starting and ending with a bun. **You've already bought \(S\) single cheeseburgers and \(D\) double cheeseburgers**. Each provides you with two buns, though a single provides one patty and one cheese, while a double provides two patties and two cheese.

{{PHOTO_ID:1367507087507489|WIDTH:600}}

You'd like to know **whether you can build a \(K\)-decker cheeseburger** with the ingredients from \(S\) single and \(D\) double cheeseburgers.

# Constraints
\(1 \leq T \leq 40\)
\(0 \leq S, D \leq 100\)
\(1 \leq K \leq 100\)

# Input Format

Input begins with an integer \(T\), the number of test cases. Each case contains one line with three space-separated integers, \(S\) and \(D\) and \(K\).

# Output Format

For the \(i\)th test case, print "`Case #i:` " followed by "`YES`" if you have enough ingredients to build a \(K\)-decker cheeseburger, or "`NO`" otherwise.

# Sample Explanation
In the first case, you have one single and one double cheeseburger. In total, you have \(4\) buns, \(3\) slices of cheese, and \(3\) patties. This gives you exactly enough ingredients to build a \(3\)-decker cheeseburger.

In the second case, you have \(4\) buns, but a \(4\)-decker cheeseburger would require \(5\), so you cannot build it.

In the third case, you have plenty of ingredients to build a \(1\)-decker cheeseburger. You'll even have \(4\) single and \(5\) double cheeseburgers left over afterwards.

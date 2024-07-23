Let's think about what ingredient is limiting us. If we've bought \(0\) singles, we're limited by buns. Otherwise we're limited by patties. So we should either:

- First reasonable option: buy only doubles (we'll be limited by buns unfortunately, but maybe singles are so outrageously expensive that it's worth it), or
- Second reasonable option: buy \(1\) single first so that we're only limited by patties, then buy as many patties as possible.

The second option is a bit trickier than it seems. Specifically, we might want to buy \(\lfloor C/A \rfloor\) singles (and no doubles) or \(1\) single (and as many doubles) **or \(2\) singles (and as many doubles)**.

We might want to buy \(2\) singles because if we start with a single, then prefer doubles, we might run out of money for doubles, but be able to afford one more single. Intuitively it might sound silly to buy \(2\) singles when \(1\) double would be cheaper, but keep in mind that for the first \(2\) (and only the first \(2\)) singles you buy, this may actually be useful because you get an extra bun. If you had bought only doubles, you'd be limited by buns, not patties.

In short, it is sufficient to consider \(S\) as one of \(\{0, 1, 2, \lfloor C/A \rfloor\}\), and then buying as many doubles as possible, and taking the best of these candidates.

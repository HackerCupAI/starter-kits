If there's only one element, we don't care what it is, a \(1\) will match it. 

Otherwise, let's solve this backwards: instead of attempting to add a number in, we'll consider starting with a working solution and removing some number from it. Let's denote the daily sum of apple weights as \(K\). We can imagine having a sorted array of \(2*N\) elements where the first and last elements sum to \(K\), the second and second last sum \(K\), etc.

Now, consider removing one element from the \(2*N\) sorted numbers. There are three cases:

1. We removed the first element. Then, \(K\) is the new first element plus the second last element.
2. We removed the last element. Then, \(K\) is the new last element plus the second element.
3. We removed some element in the middle. Then, \(K\) is the sum of the first and last elements.

With \(3\) candidates for \(K\), we'll pick the smallest that works. To check if a \(K\) works, we can use two pointers \(l\) and \(r\) moving from the outside inwards on the sorted input array. If the left and right values sum less than \(K\), we record a "skip" (noting the element that would've been removed) and increment \(l\). If it sums more, we decrement \(r\), else we advance both closer. In the end, \(K\) works if there is at most a single skip (and that removed element is positive).

Running two pointers to check one candidate \(K\) takes \(\mathcal{O}(N)\) time, so the overall running time on \(3\) candidates is still \(\mathcal{O}(N)\).

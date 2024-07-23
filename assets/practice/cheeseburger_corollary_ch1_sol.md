Each single provides \(2\) buns and \(1\) patty. Each double provides \(2\) buns and \(2\) patties. Given \(S\) singles and \(D\) doubles, we will have \(2*(S + D)\) buns and \(S + 2*D\) patties.

A \(K\)-decker happens to require \(K+1\) buns and \(K\) patties. To know if we can build one, it suffices to check that both \(2*(S + D) \ge K + 1\) and \(S + 2*D \ge K\) hold true.


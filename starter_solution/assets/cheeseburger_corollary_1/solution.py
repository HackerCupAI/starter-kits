```python
def can_build_k_decker(S, D, K):
    total_buns = 2 * (S + D)
    total_cheese = S + 2 * D
    total_patties = S + 2 * D

    needed_buns = K + 1
    needed_cheese = K
    needed_patties = K

    return total_buns >= needed_buns and total_cheese >= needed_cheese and total_patties >= needed_patties

def solve(input_data):
    lines = input_data.strip().split("\n")
    T = int(lines[0])
    results = []

    for i in range(1, T + 1):
        S, D, K = map(int, lines[i].split())
        if can_build_k_decker(S, D, K):
            results.append(f"Case #{i}: YES")
        else:
            results.append(f"Case #{i}: NO")

    return "\n".join(results)

if __name__ == "__main__":
    with open(f"assets/cheeseburger_corollary_1/2023_practice_cheeseburger_corollary_ch1.in", "r") as f:
        input_data = f.read()
    generated_solution = solve(input_data)
    with open("generated_solution.out", "w") as f:
        f.write(generated_solution)
```
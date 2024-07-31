
def solve(input_data):
    lines = input_data.strip().split('\n')
    T = int(lines[0])
    results = []

    for i in range(1, T + 1):
        S, D, K = map(int, lines[i].split())
        
        bun_count = 2 * (S + D)
        cheese_count = S + 2 * D
        patty_count = S + 2 * D

        required_buns = K + 1
        required_cheese = K
        required_patties = K

        if bun_count >= required_buns and cheese_count >= required_cheese and patty_count >= required_patties:
            results.append(f"Case #{i}: YES")
        else:
            results.append(f"Case #{i}: NO")
    
    return "\n".join(results)

if __name__ == "__main__":
    with open("assets/cheeseburger_corollary_1/2023_practice_cheeseburger_corollary_ch1.in", "r") as f:
        input_data = f.read()
    generated_solution = solve(input_data)
    with open("assets/cheeseburger_corollary_1/2023_practice_cheeseburger_corollary_ch1.out", "w") as f:
        f.write(generated_solution)

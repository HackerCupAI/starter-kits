def can_build_k_decker_cheeseburger(singles, doubles, k):
    # Calculate the total number of buns, cheese slices, and patties we have
    total_buns = 2 * (singles + doubles)
    total_cheese = singles + 2 * doubles
    total_patties = singles + 2 * doubles
    
    # Calculate the required number of buns, cheese slices, and patties for a k-decker cheeseburger
    required_buns = k + 1
    required_cheese = k
    required_patties = k
    
    # Check if we have enough of each ingredient
    if total_buns >= required_buns and total_cheese >= required_cheese and total_patties >= required_patties:
        return "YES"
    else:
        return "NO"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    t = int(data[0])
    index = 1
    
    results = []
    for i in range(1, t + 1):
        s = int(data[index])
        d = int(data[index+1])
        k = int(data[index+2])
        index += 3
        
        result = can_build_k_decker_cheeseburger(s, d, k)
        results.append(f"Case #{i}: {result}")
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
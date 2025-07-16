file = "datasets/test_6_20/scenes/scene_depth_stats.txt"

# Read the file and extract average and fused depth points for each scene
average_points = []
fused_points = []

with open(file, "r") as f:
    lines = f.readlines()

# Parse the file to extract average and fused depth points
for line in lines:
    line = line.strip()
    if "average num depth points:" in line:
        # Extract the number after the colon
        parts = line.split("average num depth points:")
        if len(parts) == 2:
            avg_points = float(parts[1].strip())
            average_points.append(avg_points)
    elif "fused num depth points:" in line:
        # Extract the fused number after the colon
        parts = line.split("fused num depth points:")
        if len(parts) == 2:
            fused_pts = float(parts[1].strip())
            fused_points.append(fused_pts)

# Calculate overall averages
if average_points:
    overall_average = sum(average_points) / len(average_points)
    print(f"Number of scenes: {len(average_points)}")
    print(f"Overall average points per scene: {overall_average:.2f}")
    print(f"Minimum average points: {min(average_points):.2f}")
    print(f"Maximum average points: {max(average_points):.2f}")

    # Print individual scene averages for reference
    print("\nIndividual scene averages:")
    for i, avg in enumerate(average_points, 1):
        print(f"Scene {i}: {avg:.2f} points")
else:
    print("No average depth points data found in the file")

# Calculate overall fused averages
if fused_points:
    overall_fused_average = sum(fused_points) / len(fused_points)
    print(f"\n--- Fused Depth Points Statistics ---")
    print(f"Number of scenes with fused data: {len(fused_points)}")
    print(f"Overall fused average points per scene: {overall_fused_average:.2f}")
    print(f"Minimum fused points: {min(fused_points):.2f}")
    print(f"Maximum fused points: {max(fused_points):.2f}")

    # Print individual scene fused points for reference
    print("\nIndividual scene fused points:")
    for i, fused in enumerate(fused_points, 1):
        print(f"Scene {i}: {fused:.2f} points")
else:
    print("No fused depth points data found in the file")

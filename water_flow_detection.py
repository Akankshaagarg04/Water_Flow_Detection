import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from statistics import mode

# video_path = r"C:\Users\Akanksha\OneDrive\Desktop\turby water.mp4"  # turby
video_path = r"C:\Users\Akanksha\OneDrive\Desktop\laminar water.mp4"  # laminar

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))

with open("flow_analysis.txt", "w") as output_file:
    output_file.write("Time(s)\tEdge Density\tEdge Orientation StdDev\tFlow Type\n")

    time_stamps = []
    edge_densities = []
    orientation_stds = []
    flow_types = []

    start_time = time.time()
    frame_count = 0
    while(1):  
        ret, frame = cap.read()
        
        if not ret:
            break

        if frame_count % fps == 0:
            current_time = time.time() - start_time

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            edges = cv2.Canny(blurred, threshold1=40, threshold2=100)

            total_pixels = edges.shape[0] * edges.shape[1]
            edge_pixels = cv2.countNonZero(edges)
            edge_density = edge_pixels / total_pixels

            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

            if lines is not None:
                orientations = []
                for line in lines:
                    rho, theta = line[0]
                    orientations.append(np.degrees(theta))  
                
                edge_orientation_std = np.std(orientations)
            else:
                edge_orientation_std = 90

            if edge_density < 0.05:
                flow_type = "Slow Flow"
            elif edge_density < 0.15:
                if edge_orientation_std < 15:
                    flow_type = "Laminar Flow"
                else:
                    flow_type = "Medium Flow"
            else:
                if edge_orientation_std < 15:
                    flow_type = "Fast Laminar Flow"
                else:
                    flow_type = "Turbulent Flow"

            
            time_stamps.append(current_time)
            edge_densities.append(edge_density)
            orientation_stds.append(edge_orientation_std)
            flow_types.append(flow_type)

            print(f"Edge Density: {edge_density:.4f}")
            print(f"Edge Orientation Standard Deviation: {edge_orientation_std:.2f}")
            print(f"Water Flow Classification: {flow_type}")
            
            
            output_file.write(f"{int(current_time)}\t{edge_density:.4f}\t{edge_orientation_std:.2f}\t{flow_type}\n")
            
           
            cv2.imshow("Original Frame", frame)
            cv2.imshow("Edges", edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

cap.release()
cv2.destroyAllWindows()

if flow_types:
    flow_mode = mode(flow_types)
    print(f"\nMode of Water Flow Classification: {flow_mode}")
else:
    print("\nNo water flow classifications found.")

print("Analysis complete")

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time_stamps, edge_densities, label='Edge Density', color='b')
plt.title("Edge Density Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Edge Density")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_stamps, orientation_stds, label='Edge Orientation Std Dev', color='g')
plt.title("Edge Orientation Std Dev Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Orientation Std Dev")
plt.grid(True)

plt.tight_layout()
plt.show()

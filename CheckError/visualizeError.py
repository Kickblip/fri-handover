"""
Interactive WebGL visualization comparing predicted vs actual hand coordinates.
Generates an HTML file with 2D trajectory plots and similarity metrics.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

def load_predictions(pred_file):
    """Load prediction CSV and return as dict keyed by (frame, future_idx)."""
    df = pd.read_csv(pred_file)
    predictions = {}
    
    for _, row in df.iterrows():
        frame = int(row['frame'])
        future_idx = int(row['future_frame_idx'])
        actual_frame = frame + 1 + future_idx  # Map to actual frame number
        
        # Extract 21 landmarks (x, y, z)
        coords = []
        for lm in range(21):
            x = row[f'lm_{lm}_x']
            y = row[f'lm_{lm}_y']
            z = row[f'lm_{lm}_z']
            coords.append([x, y, z])
        
        predictions[(frame, future_idx, actual_frame)] = np.array(coords)
    
    return predictions

def load_actuals(actual_file):
    """Load actual receiving hand (hand_0) coordinates from CSV."""
    df = pd.read_csv(actual_file)
    actuals = {}
    
    for _, row in df.iterrows():
        frame = int(row['frame_idx'])
        
        # Extract hand_0 (h0) landmarks - receiving hand
        coords = []
        for lm in range(21):
            x_col = f'h0_lm{lm}_x'
            y_col = f'h0_lm{lm}_y'
            z_col = f'h0_lm{lm}_z'
            
            if x_col in row and pd.notna(row[x_col]):
                x = row[x_col]
                y = row[y_col]
                z = row[z_col]
                coords.append([x, y, z])
            else:
                coords.append([np.nan, np.nan, np.nan])
        
        actuals[frame] = np.array(coords)
    
    return actuals

def compute_metrics(predictions, actuals):
    """Compute similarity metrics between predictions and actuals."""
    results = []
    
    for (pred_frame, future_idx, actual_frame), pred_coords in predictions.items():
        if actual_frame not in actuals:
            continue
        
        actual_coords = actuals[actual_frame]
        
        # Skip if actual has NaN
        if np.any(np.isnan(actual_coords)):
            continue
        
        # Compute per-landmark distance
        distances = np.linalg.norm(pred_coords - actual_coords, axis=1)
        
        # Compute metrics
        mse = np.mean(distances ** 2)
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        
        results.append({
            'pred_frame': pred_frame,
            'future_idx': future_idx,
            'actual_frame': actual_frame,
            'mse': mse,
            'mean_distance': mean_distance,
            'max_distance': max_distance,
            'pred_coords': pred_coords.tolist(),
            'actual_coords': actual_coords.tolist(),
            'distances': distances.tolist()
        })
    
    return results

def generate_html(results, output_file):
    """Generate interactive HTML visualization with WebGL/Plotly."""
    
    # Prepare data for plotting
    frames = [r['actual_frame'] for r in results]
    mse_values = [r['mse'] for r in results]
    mean_distances = [r['mean_distance'] for r in results]
    max_distances = [r['max_distance'] for r in results]
    
    # Compute overall statistics
    overall_mse = np.mean(mse_values)
    overall_mean_dist = np.mean(mean_distances)
    overall_max_dist = np.max(max_distances)
    
    # Prepare trajectory data (wrist landmark - lm_0)
    pred_wrist_x = [r['pred_coords'][0][0] for r in results]
    pred_wrist_y = [r['pred_coords'][0][1] for r in results]
    pred_wrist_z = [r['pred_coords'][0][2] for r in results]
    
    actual_wrist_x = [r['actual_coords'][0][0] for r in results]
    actual_wrist_y = [r['actual_coords'][0][1] for r in results]
    actual_wrist_z = [r['actual_coords'][0][2] for r in results]
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Hand Coordinate Prediction Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .plot-container {{
            margin: 20px 0;
        }}
        .info {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Hand Coordinate Prediction vs Actual Analysis</h1>
        
        <div class="info">
            <strong>Dataset Info:</strong> {len(results)} prediction-actual pairs analyzed<br>
            <strong>Landmarks:</strong> 21 hand landmarks per frame<br>
            <strong>Metrics:</strong> MSE (Mean Squared Error), Mean Distance, Max Distance (in coordinate units)
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Overall MSE</div>
                <div class="stat-value">{overall_mse:.6f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Distance</div>
                <div class="stat-value">{overall_mean_dist:.6f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Distance</div>
                <div class="stat-value">{overall_max_dist:.6f}</div>
            </div>
        </div>
        
        <h2>Error Metrics Over Time</h2>
        <div id="error-plot" class="plot-container"></div>
        
        <h2>Wrist Trajectory: Predicted vs Actual (X-Y Plane)</h2>
        <div id="trajectory-xy" class="plot-container"></div>
        
        <h2>Wrist Trajectory: Predicted vs Actual (X-Z Plane)</h2>
        <div id="trajectory-xz" class="plot-container"></div>
        
        <h2>3D Wrist Trajectory</h2>
        <div id="trajectory-3d" class="plot-container"></div>
        
        <h2>Per-Landmark Mean Distance Heatmap</h2>
        <div id="landmark-heatmap" class="plot-container"></div>
    </div>
    
    <script>
        // Error metrics plot
        const errorData = [
            {{
                x: {json.dumps(frames)},
                y: {json.dumps(mse_values)},
                name: 'MSE',
                type: 'scatter',
                mode: 'lines+markers',
                line: {{color: '#ff6b6b'}}
            }},
            {{
                x: {json.dumps(frames)},
                y: {json.dumps(mean_distances)},
                name: 'Mean Distance',
                type: 'scatter',
                mode: 'lines+markers',
                line: {{color: '#4ecdc4'}},
                yaxis: 'y2'
            }}
        ];
        
        const errorLayout = {{
            title: 'Prediction Error Across Frames',
            xaxis: {{title: 'Frame Number'}},
            yaxis: {{title: 'MSE', side: 'left', color: '#ff6b6b'}},
            yaxis2: {{
                title: 'Mean Distance',
                side: 'right',
                overlaying: 'y',
                color: '#4ecdc4'
            }},
            hovermode: 'x unified',
            height: 400
        }};
        
        Plotly.newPlot('error-plot', errorData, errorLayout, {{responsive: true}});
        
        // X-Y trajectory
        const trajectoryXYData = [
            {{
                x: {json.dumps(actual_wrist_x)},
                y: {json.dumps(actual_wrist_y)},
                name: 'Actual',
                type: 'scatter',
                mode: 'lines+markers',
                marker: {{size: 6, color: '#2ecc71'}},
                line: {{color: '#2ecc71', width: 2}}
            }},
            {{
                x: {json.dumps(pred_wrist_x)},
                y: {json.dumps(pred_wrist_y)},
                name: 'Predicted',
                type: 'scatter',
                mode: 'lines+markers',
                marker: {{size: 6, color: '#e74c3c'}},
                line: {{color: '#e74c3c', width: 2, dash: 'dot'}}
            }}
        ];
        
        const trajectoryXYLayout = {{
            title: 'Wrist Position (X-Y Plane)',
            xaxis: {{title: 'X Coordinate'}},
            yaxis: {{title: 'Y Coordinate'}},
            hovermode: 'closest',
            height: 500
        }};
        
        Plotly.newPlot('trajectory-xy', trajectoryXYData, trajectoryXYLayout, {{responsive: true}});
        
        // X-Z trajectory
        const trajectoryXZData = [
            {{
                x: {json.dumps(actual_wrist_x)},
                y: {json.dumps(actual_wrist_z)},
                name: 'Actual',
                type: 'scatter',
                mode: 'lines+markers',
                marker: {{size: 6, color: '#2ecc71'}},
                line: {{color: '#2ecc71', width: 2}}
            }},
            {{
                x: {json.dumps(pred_wrist_x)},
                y: {json.dumps(pred_wrist_z)},
                name: 'Predicted',
                type: 'scatter',
                mode: 'lines+markers',
                marker: {{size: 6, color: '#e74c3c'}},
                line: {{color: '#e74c3c', width: 2, dash: 'dot'}}
            }}
        ];
        
        const trajectoryXZLayout = {{
            title: 'Wrist Position (X-Z Plane)',
            xaxis: {{title: 'X Coordinate'}},
            yaxis: {{title: 'Z Coordinate (Depth)'}},
            hovermode: 'closest',
            height: 500
        }};
        
        Plotly.newPlot('trajectory-xz', trajectoryXZData, trajectoryXZLayout, {{responsive: true}});
        
        // 3D trajectory
        const trajectory3DData = [
            {{
                x: {json.dumps(actual_wrist_x)},
                y: {json.dumps(actual_wrist_y)},
                z: {json.dumps(actual_wrist_z)},
                name: 'Actual',
                type: 'scatter3d',
                mode: 'lines+markers',
                marker: {{size: 4, color: '#2ecc71'}},
                line: {{color: '#2ecc71', width: 4}}
            }},
            {{
                x: {json.dumps(pred_wrist_x)},
                y: {json.dumps(pred_wrist_y)},
                z: {json.dumps(pred_wrist_z)},
                name: 'Predicted',
                type: 'scatter3d',
                mode: 'lines+markers',
                marker: {{size: 4, color: '#e74c3c'}},
                line: {{color: '#e74c3c', width: 4}}
            }}
        ];
        
        const trajectory3DLayout = {{
            title: 'Wrist 3D Trajectory',
            scene: {{
                xaxis: {{title: 'X'}},
                yaxis: {{title: 'Y'}},
                zaxis: {{title: 'Z (Depth)'}}
            }},
            height: 600
        }};
        
        Plotly.newPlot('trajectory-3d', trajectory3DData, trajectory3DLayout, {{responsive: true}});
        
        // Per-landmark heatmap
        const landmarkDistances = [];
        for (let lm = 0; lm < 21; lm++) {{
            const distances = {json.dumps([r['distances'] for r in results])};
            const lmDists = distances.map(d => d[lm]);
            landmarkDistances.push(lmDists);
        }}
        
        const heatmapData = [{{
            z: landmarkDistances,
            x: {json.dumps(frames)},
            y: Array.from({{length: 21}}, (_, i) => `Landmark ${{i}}`),
            type: 'heatmap',
            colorscale: 'Viridis',
            colorbar: {{title: 'Distance'}}
        }}];
        
        const heatmapLayout = {{
            title: 'Per-Landmark Distance Across Frames',
            xaxis: {{title: 'Frame Number'}},
            yaxis: {{title: 'Landmark ID'}},
            height: 600
        }};
        
        Plotly.newPlot('landmark-heatmap', heatmapData, heatmapLayout, {{responsive: true}});
    </script>
</body>
</html>'''
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Generated visualization: {output_file}")
    print(f"📊 Overall MSE: {overall_mse:.6f}")
    print(f"📏 Mean Distance: {overall_mean_dist:.6f}")
    print(f"📈 Max Distance: {overall_max_dist:.6f}")
    print(f"📦 Analyzed {len(results)} prediction-actual pairs")

def main():
    # File paths
    script_dir = Path(__file__).parent
    pred_file = script_dir / "40_video_future_predictions (1).csv"
    actual_file = script_dir / "40_video_hands.csv"
    output_file = script_dir / "prediction_analysis.html"
    
    print("🔍 Loading predictions...")
    predictions = load_predictions(pred_file)
    
    print("🔍 Loading actuals...")
    actuals = load_actuals(actual_file)
    
    print("📊 Computing metrics...")
    results = compute_metrics(predictions, actuals)
    
    if not results:
        print("❌ No matching prediction-actual pairs found!")
        return
    
    print("🎨 Generating visualization...")
    generate_html(results, output_file)
    
    print(f"\n✨ Done! Open {output_file} in your browser to view the analysis.")

if __name__ == "__main__":
    main()

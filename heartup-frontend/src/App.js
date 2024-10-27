import './App.css';
import { useState } from 'react';

function App() {
  const [selectedDataset, setSelectedDataset] = useState('');
  const [visualizationData, setVisualizationData] = useState('');
  const [analyticsData, setAnalyticsData] = useState('');

  const datasets = [
    { name: 'Patient 1', value: 'dataset1' },
    { name: 'Patient 2', value: 'dataset2' }
  ];

  const handleDatasetChange = (event) => {
    const dataset = event.target.value;
    setSelectedDataset(dataset);

    if (dataset === 'dataset1') {
      setVisualizationData('Visualization data for Dataset 1');
      setAnalyticsData('Predictive analytics for Dataset 1');
    } else if (dataset === 'dataset2') {
      setVisualizationData('Visualization data for Dataset 2');
      setAnalyticsData('Predictive analytics for Dataset 2');
    } else {
      setVisualizationData('');
      setAnalyticsData('');
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>HeartUp</h1>
      </header>
      <div className="container" style={{ display: 'flex', height: '80vh' }}>
        <div className="dataset-selection" style={{ flex: '1', padding: '20px' }}>
          <h2>Select Patient</h2>
          <p>Data recorded from ECG test</p>
          <select
            value={selectedDataset}
            onChange={handleDatasetChange}
            style={{
              marginBottom: '20px',
              padding: '10px',
              fontSize: '16px',
              borderRadius: '5px',
              border: '1px solid #ccc',
            }}
            onFocus={(e) => (e.target.style.borderColor = '#e63946')}
            onBlur={(e) => (e.target.style.borderColor = '#ccc')}
          >
            <option value="">-- Select a dataset --</option>
            {datasets.map((dataset) => (
              <option key={dataset.value} value={dataset.value}>
                {dataset.name}
              </option>
            ))}
          </select>
        </div>
        <div className="visualization" style={{ flex: '2', padding: '20px', borderLeft: '1px solid #ccc' }}>
          <h2>Cardiac Digital Twin</h2>
          <p>Visualizing heart rhythm with ECG data</p>
          <div className="chart">
            <p>{visualizationData || '[ Select a dataset to display visualization ]'}</p>
          </div>
        </div>
        <div className="analytics" style={{ flex: '2', padding: '20px', borderLeft: '1px solid #ccc' }}>
          <h2>Predictive Analytics</h2>
          <div className="analytics-content">
            <p>{analyticsData || '[ Select a dataset to display predictive analytics ]'}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
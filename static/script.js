function submitForm() {
    const form = document.getElementById('diagnosticsForm');
    const formData = new FormData(form);

    // Convert FormData to a plain object
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });

    // Basic client-side validation
    if (!data.name || !data.age || !data.gender || !data.bmi || !data.glucose) {
        document.getElementById('result').innerText = 'Please fill in all required fields.';
        return;
    }

    // Show loading state
    document.getElementById('result').innerText = 'Processing... Please wait.';
    console.log('Sending data:', data);

    // Send data to the backend as a POST request
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Response data:', data);

        if (data.prediction !== undefined) {
            // Display the prediction result
            document.getElementById('result').innerText = `Prediction: ${data.prediction}`;
        } else if (data.error) {
            // Display error message from the server
            document.getElementById('result').innerText = `Error: ${data.error}`;
        } else {
            document.getElementById('result').innerText = 'Unexpected response from the server.';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'An error occurred while processing your request.';
    });
}

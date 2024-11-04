async function predictDiabetes() {
    const age = document.getElementById("age").value;
    const glucose = document.getElementById("glucose").value;
    const insulin = document.getElementById("insulin").value;
    const bmi = document.getElementById("bmi").value;
    const modelType = document.getElementById("model").value;
  
    const data = {
      age: Number(age),
      glucose: Number(glucose),
      insulin: Number(insulin),
      bmi: Number(bmi),
      model_type: modelType
    };
  
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
      });
  
      const result = await response.json();
  
      if (response.ok) {
        document.getElementById("result").textContent = `Diabetes Type Prediction: ${result.diabetes_type}`;
      } else {
        document.getElementById("result").textContent = `Error: ${result.error}`;
      }
    } catch (error) {
      document.getElementById("result").textContent = `Error: ${error.message}`;
    }
  }
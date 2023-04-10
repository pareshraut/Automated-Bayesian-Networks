// create a react component which inputs a textarea message  then perforsm a fetch request to localhost:3001 gets back a response as data.message and displays it in a box below
import React, { useState } from 'react';
import './App.css';

function App() { 
  const [message, setMessage] = useState(''); 
  const [response, setResponse] = useState('');

  const handleSubmit = (e) => { 
    e.preventDefault(); 
    fetch('http://localhost:3001/', { 
      method: 'POST', 
      headers: { 
        'Content-Type': 'application/json', 
      },
      body: JSON.stringify({ message }), 
    })
    .then((res) => res.json())
    .then((data) => setResponse(data.message));
  };

  return (
    <div className="App">
      <h1>OpenAI Bayesian risk manager Chatbot</h1>
      <form onSubmit={handleSubmit}>
          <textarea 
            value={message} 
            placeholder="What specific aspects of the risk scenario 'market crash in technology market' are you interested in modeling?
                         Please provide any relevant details or factors that you think might be important"
            onChange={(e) => setMessage(e.target.value)}
            style={{ height: '200px', width: '80%' }}
            ></textarea>               
        <button type="submit">Submit</button>
      </form>
      {response && <div><b>Bayesiachat:</b>{response}</div>}
    </div>
  );
}


export default App;
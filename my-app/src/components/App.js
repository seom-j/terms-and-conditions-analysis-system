import React, { useState } from 'react';
import './App.css';
import InputComponent from './UI/InputComponent';
import OutputComponent from './UI/OutputComponent';

function App() {
  const [userInput, setUserInput] = useState('');
  const [selectedValue, setSelectedValue] = useState('');
  const [receiveData, setReceiveData] = useState({positive:[],negative:[],keywords:[]});
  const [update, setUpdate] = useState(false);

  console.log("ReceiveData in App:", receiveData);  
  console.log("SelectedValue in App:", selectedValue);  
  console.log("update in App:", update);  
  
  const handleInputChange = (e) => {
    setUserInput(e.target.value);
  };
  
  return (
    <div className="App">
      <h1 id='title'>AI 약관 상세 분석 시스템 {selectedValue}</h1>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr 1fr' }}>
      <div className='remain'>
      
      </div>
      <div className="container">
        <InputComponent handleInputChange={handleInputChange}  receiveData={receiveData} selectedValue={selectedValue} setUpdate={setUpdate}/>
        <OutputComponent userInput={userInput} setSelectedValue={setSelectedValue}  setReceiveData={setReceiveData} update={update} setUpdate={setUpdate}/>
      </div>
      <div className='remain'></div>
      </div>

    </div>
  );
}

export default App;
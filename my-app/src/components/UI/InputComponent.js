import React, { useState } from 'react';
import './InputComponent.css';
import Tasks from './Task.js';

function InputComponent({ selectedValue, receiveData, setUpdate }) {
  const [extractedText, setExtractedText] = useState('');
  const [resultText, setResultText] = useState([]);
  const [showTextArea, setShowTextArea] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // fetch.api 사용 코드 안돼면 주석처리하고 위에 주석을 해제하고 사용하면 됌
  const handleImageUpload = async (e) => {
    const image = e.target.files[0];
    setExtractedText('');
    setShowTextArea(true);
    try {
      const formData = new FormData();
      formData.append('image', image);

      const url = 'http://localhost:5000/api/upload';
      const response = await fetch(url, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      //const concatenatedText = extractedText + data.text;
      setExtractedText(data.text);
      console.log('Upload successful:', data);

    } catch (error) {
      console.error('Error during image upload:', error);
    }
  };

  const handleServerSend = async () => {
    setIsLoading(true);
    setError(null);
    setUpdate(true);
    setShowTextArea(false);
    try {
      const sentences = extractedText.split(/(?<!\d)[.\n?!]\s*/).filter(sentence => sentence.trim() !== '').map(sentence => {
        if (sentence.endsWith('다')) {
          return sentence + '.';
        }
        return sentence;
      });

      const url = 'http://localhost:5000/api/process';
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ sentences })
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      console.log('Server send successful:', data);
      setResultText(data.text);
      console.log('resultdat', data.text);

    } catch (error) {
      console.error('Error during sending to server:', error);

    }
    setIsLoading(false);

  };
  // 보기 상태를 변경하는 버튼 클릭 핸들러
  const handleViewChange = () => {
    setShowTextArea(!showTextArea);
  };
  return (
    <div className="input-component">
      {showTextArea ? (
        <textarea
          className="input-text"
          placeholder="약관을 입력하세요..."
          value={extractedText}
          onChange={e => setExtractedText(e.target.value)}
          style={{ whiteSpace: 'pre-wrap' }}
        />) : (
        <Tasks
          items={resultText}
          loading={isLoading}
          error={error}
          onFetch={handleServerSend}
          receiveData={receiveData}
          selectedValue={selectedValue}
        ></Tasks>
      )}
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <label htmlFor="file-upload" className="file-upload-button">
          <span>파일 선택</span>
          {/* <img src="https://cdn-icons-png.flaticon.com/128/1828/1828470.png" width="20" alt="upload-file" /> */}
          <img src="https://cdn-icons-png.flaticon.com/512/1086/1086933.png" width="20" alt="upload-file" style={{ marginLeft: '5px' }} /></label>
        <input id="file-upload" type="file" accept="image/*" onChange={handleImageUpload} style={{ display: 'none' }} />
        <div style={{ display: 'flex', flexWrap: 'wrap' }}>
          <button className='submit' onClick={handleServerSend}>서버로 전송</button>
          <button className='change' onClick={handleViewChange}>
            {showTextArea ? '결과 보기' : '편집하기'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default InputComponent;
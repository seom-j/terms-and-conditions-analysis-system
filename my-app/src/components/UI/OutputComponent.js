import React, { useState } from 'react';
import './OutputComponent.css';


function OutputComponent({ setSelectedValue, setReceiveData, update, setUpdate  }) {
  const [selectedOption, setSelectedOption] = useState('');
  const [pnData, setpnData] = useState({
    P_Text: [],
    N_Text: [],
    Sum_P: [],
    Sum_N: [],
    T_Text: [],
    K_Text: []
  });


  async function fetchDataHandler() {
    try {
      const response = await fetch('http://localhost:5000/process');
      if (!response.ok) {
        // 요청이 실패한 경우
        throw new Error('Failed to fetch data');
      }

      const data = await response.json();
      if (!data.positive_texts || !data.negative_texts || !data.sum_p_texts || !data.sum_n_texts) {
        // 데이터 형식이 잘못된 경우
        throw new Error('Invalid data format');
      }
      const transformedData = {
        P_Text: data.positive_texts,
        N_Text: data.negative_texts,
        Sum_P: data.sum_p_texts,
        Sum_N: data.sum_n_texts,
        T_Text: data.title_texts,
        K_Text: data.keyword_texts[0]
      };

      setpnData(transformedData);
      setReceiveData({ // 기존 코드 수정
        positive: data.positive_texts,
        negative: data.negative_texts,
        keywords: data.keyword_texts[0]
      });
    }
    catch (error) {
      // 오류 처리: 응답이 없거나 데이터 형식이 잘못된 경우
      console.error(error);
      // 팝업 또는 다른 방식으로 오류 메시지를 사용자에게 보여줄 수 있습니다.
    }
  }


  let output;

  const handleProcessClick = (option) => {
    if (option === selectedOption) {
      setSelectedOption('');
      setSelectedValue('');
    } else {
      setSelectedOption(option);
      setSelectedValue(option);
    }
  };


  if (selectedOption === ": 유리 / 불리 판단") {
    output = (
      <div className="output-box">
        <div className="judgement-box">
          <h3 style={{color: 'blue'}}>유리한 조건</h3>
          <ul>
            {pnData.Sum_P.map((text, index) => (
              <li key={index}>{text}</li>
            ))}
          </ul>
        </div>
        <div className="summary-box">
          <h3 style={{color: 'red'}}>불리한 조건</h3>
          <ul>
            {pnData.Sum_N.map((text, index) => (
              <li key={index}>{text}</li>
            ))}
          </ul>
        </div>
      </div>
    );
  }
  else if (selectedOption === ": 키워드 분석") {
    output = (
      <div className="output-box">
        <div className="title-box">
          <h3>타이틀</h3>
          {<ul>
            {pnData.T_Text.map((text, index) => (
              <li key={index}>{text}</li>
            ))}
          </ul>}
        </div>

        <div className="keyword-box">
          <h3>키워드</h3>
          {<ul>
            {pnData.K_Text.map((text, index) => (
              <li key={index}>{text}</li>
            ))}
          </ul>}
        </div>
      </div>
    );
  }
  else {
    output = <div className='output-box'>
      <p>
        1. 분석을 진행할 <b>약관이미지를 선택</b> or <b>텍스트를 입력</b>해주세요. <br></br>
      2. 약관 이미지 추출시 추출된 <b>텍스트의 정확도</b>가 낮으면 텍스트를 수정해 주세요. <br></br>
      3. <b>서버로 전송</b>을 누르면 약관 분석이 시작됩니다. <br></br>
      4. 약관 분석이 완료되면 우측의  <b>옵션</b>을 선택해  <b>키워드</b>와 <b>유불리 여부</b>를 확인할 수 있습니다.
      </p>
    </div>
  };

  return (
    <div className="output-component">
      <div className="output-buttons">
        <select
          value={selectedOption}
          onChange={(e) => { handleProcessClick(e.target.value); fetchDataHandler(); }}
        >
          <option value="">시스템을 선택하세요.</option>
          <option value=": 유리 / 불리 판단">유리 / 불리 판단</option>
          <option value=": 키워드 분석">키워드 분석</option>
        </select>
      </div>

      {output}

    </div>
  );
}

export default OutputComponent;
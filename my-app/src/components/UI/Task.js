import React from "react";
import '../UI/InputComponent.css';
import ReactDOMServer from 'react-dom/server';


const Tasks = (props) => {
  let taskList = <h2>이미지 파일을 등록하고 서버 전송을 해주세요</h2>
  console.log("ReceiveData in Tasks:", props.receiveData);
  console.log("items in App:", props.items);  // 로그 추가
  function highlightKeywords(sentence, keywords) {
    function highlightSentence(sentence, keywords) {
      let highlightedSentence = sentence;

      keywords.forEach(keyword => {
        // 키워드를 찾아서 스타일링한 후 문자열 형태로 렌더링
        const highlightedKeyword = ReactDOMServer.renderToString(
          <span key={keyword} style={{ fontWeight: 'bold', color: 'red' }}>
            {keyword}
          </span>
        );
        
        highlightedSentence = highlightedSentence.split(keyword).join(highlightedKeyword);
      });

      return highlightedSentence;
    }

    return (
      <div>
        {sentence.map((s, index) => (
          <p key={index} dangerouslySetInnerHTML={{ __html: highlightSentence(s, keywords) }} />
        ))}
      </div>
    );
  }
  

  if (Array.isArray(props.items) && props.items.length > 0) {

    if (props.selectedValue === ": 유리 / 불리 판단") {
      taskList = (
        props.items.map((text, index) => (
          <p key={index}>{text}</p>
        ))
      );
      taskList = props.items.map((text, index) => {
        let itemClass = "item-normal";

        if (props.receiveData.positive.includes(text)) {
          itemClass = "item-positive";
        } else if (props.receiveData.negative.includes(text)) {
          itemClass = "item-negative";
        } else {
          itemClass = "item-normal";
        }

        return (
          <p key={index} className={itemClass}>
            {text}
          </p>
        );
      });
    } else if (props.selectedValue === ": 키워드 분석") {
              taskList =  highlightKeywords(props.items, props.receiveData.keywords);
      }
    else {
      taskList = (
        props.items.map((text, index) => (
          <p key={index}>{text}</p>
        ))
      );
      taskList = props.items.map((text, index) => {
        let itemClass = "item-normal";


        return (
          <p key={index} className={itemClass}>
            {text}
          </p>
        );
      });
    }
  }
  let content = taskList;
  console.log("Items length:", props.items.length); // 로그를 찍어보기
  if (props.error) {
    content = <h1>에러가 발생했습니다. 이미지 파일을 등록하고 서버 전송을 해주세요</h1>;
  }
  if (props.loading) {
    content = '약관을 분석중입니다!!.. 잠시만 기다려 주세요';
  }
  return (
    <div className="input-text">{content}</div>
  )
};

export default Tasks;
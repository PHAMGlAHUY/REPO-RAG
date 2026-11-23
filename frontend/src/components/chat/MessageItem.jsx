import MessageContent from './MessageContent';

export default function MessageItem({ message, pending }) {
  const isUser = message.role === 'user';
  const waiting = pending && !message.content;

  return (
    <li className={`msg ${isUser ? 'msg--user' : 'msg--bot'} ${message.error ? 'msg--error' : ''}`}>
      {!isUser && <span className="msg__avatar" aria-hidden="true">AI</span>}
      <div className="msg__body">
        {waiting ? (
          <span className="typing" role="status" aria-label="Đang trả lời">
            <i /><i /><i />
          </span>
        ) : (
          <MessageContent text={message.content} />
        )}
      </div>
    </li>
  );
}

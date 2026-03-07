const chatBox = document.getElementById("chat-box");
const sendBtn = document.getElementById("send-btn");
const input = document.getElementById("user-input");
const newChatBtn = document.querySelector(".new-chat");


sendBtn.addEventListener("click", sendMessage);

input.addEventListener("keydown", function(e){
if(e.key === "Enter" && !e.shiftKey){
e.preventDefault();
sendMessage();
}
});

newChatBtn.addEventListener("click", newChat);


function addMessage(text, sender){

const message = document.createElement("div");

message.classList.add("message", sender);

message.innerText = text;

chatBox.appendChild(message);

chatBox.scrollTop = chatBox.scrollHeight;

}


function showTyping(){

const typing = document.createElement("div");

typing.classList.add("message","bot");

typing.innerText = "Thinking...";

typing.id = "typing";

chatBox.appendChild(typing);

}


function removeTyping(){

const typing = document.getElementById("typing");

if(typing) typing.remove();

}


async function sendMessage(){

const message = input.value.trim();

if(!message) return;

addMessage(message,"user");

input.value="";

showTyping();

try{

const response = await fetch("/chat",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({message:message})
});

const data = await response.json();

removeTyping();

addMessage(data.response,"bot");

}catch(err){

removeTyping();

addMessage("⚠️ Error connecting to server","bot");

}

}


function newChat(){

chatBox.innerHTML="";

fetch("/reset",{
method:"POST"
});

}


async function loadHistory(){

const response = await fetch("/history");

const data = await response.json();

const history = data.history;

chatBox.innerHTML="";

history.forEach(msg => {

if(msg.role === "user"){
addMessage(msg.content,"user");
}
else{
addMessage(msg.content,"bot");
}

});

}


window.onload = loadHistory;
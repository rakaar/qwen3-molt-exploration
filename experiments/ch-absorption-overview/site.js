"use strict";
const data=JSON.parse(document.getElementById("report-data").textContent);
const notes={upper:"P(CH) is measured at the first answer token. The full greedy answer is CH in all four conditions.",lower:"P( ch | child:) is measured after the exact native output prefix in a separate teacher-forced replay. Every full greedy answer remains child: ch. The generated child token is unedited."};
const body=document.getElementById("joint-rows"),note=document.getElementById("task-note"),heading=document.getElementById("probability-label");
function selectTask(task){
  body.replaceChildren();
  for(const row of data.taskrows[task]){
    const tr=document.createElement("tr");if(row.condition==="joint")tr.className="joint-row";
    for(const [i,text] of [row.label,row.answer,(row.target_probability*100).toFixed(task==="upper"?2:4)+"%"].entries()){
      const td=document.createElement("td");if(i===1){const code=document.createElement("code");code.textContent=text;td.append(code);}else td.textContent=text;tr.append(td);
    }body.append(tr);
  }
  note.textContent=notes[task];heading.textContent=task==="upper"?"P(CH)":"P( ch | child:)";
  document.getElementById("joint-panel").setAttribute("aria-labelledby",task+"-tab");
  document.querySelectorAll("[data-task]").forEach(b=>{const selected=b.dataset.task===task;b.setAttribute("aria-selected",String(selected));b.tabIndex=selected?0:-1;});
}
document.querySelectorAll("[data-task]").forEach(button=>{
  button.addEventListener("click",()=>selectTask(button.dataset.task));
  button.addEventListener("keydown",event=>{if(["ArrowLeft","ArrowRight","Home","End"].includes(event.key)){event.preventDefault();const task=event.key==="Home"?"upper":event.key==="End"?"lower":button.dataset.task==="upper"?"lower":"upper";selectTask(task);document.querySelector(`[data-task="${task}"]`).focus();}});
});
selectTask("upper");

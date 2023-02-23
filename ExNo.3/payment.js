const number1 =
document.getElementById('number1'); 
const name =document.getElementById('name'); 
const number2 =document.getElementById('number2');
const type = document.getElementById('type');
form.addEventListener('submit', e => 
{
e.preventDefault(); 
    checkInputs();
});
function checkInputs() { // trim to remove thewhitespaces 
    const number1Value =number1.value.trim(); 
    const nameValue =name.value.trim();
    const number2Value =number2.value.trim();
const typeValue = type.value.trim();
if(number1Value.length !== 16 || number1Value.length === '')
{

setErrorFor(number1, 'invalid card number');
}else{
setSuccessFor(number1);
}
if(number2Value.length !== 3 || number2Value.length === '')
{
setErrorFor(number2, 'invalid ccv number');
}else{
setSuccessFor(number2);
}
if(nameValue === '') {
setErrorFor(name, 'card name cannot be blank');
} else {
setSuccessFor(name);
}
if(typeValue === '')
{
setErrorFor(type, 'invalid card type');
}
else {
setSuccessFor(type);
}
}
function setErrorFor(input, message) {
const formControl = input.parentElement;
const small =
formControl.querySelector('small');
formControl.className = 'form-controlerror'; 
small.innerText = message;

}
function setSuccessFor(input) { const formControl
= input.parentElement; formControl.className =
'form-control success';
}

//Variables for each item const item1 =
document.querySelector("#item1"); const item2 =
document.querySelector("#item2"); const item3 =
document.querySelector("#item3"); const item4 =
document.querySelector("#item4"); const item5 =
document.querySelector("#item5"); const item6 =
document.querySelector("#item6");
//Submit button
const subbutton = document.querySelector("#subbutton");
//Total cost of shoppin cart
let total = 0.0;
//Value of each item
item1.value = 500;
item2.value = 200;
item3.value = 75; item4.value
= 350; item5.value = 70;
item6.value = 100;

//Making sure user has selected atleast one item subbutton.addEventListener("click",
() => {
if (!(item1.checked || item2.checked || item3.checked || item4.checked || item5.checked ||
item6.checked)) { alert("Failure");
} else {
alert("Success");
}
//Adding price of each selected item
if (item1.checked) total += parseFloat(item1.value);
if (item2.checked) total += parseFloat(item2.value);
if (item3.checked) total += parseFloat(item3.value);
if (item4.checked) total += parseFloat(item4.value);
if (item5.checked) total += parseFloat(item5.value);
if (item6.checked) total += parseFloat(item6.value);

//Displaying total price
alert("Total: " + total); //Storing total price
localStorage.setItem("total", total);
};
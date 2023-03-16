const fname = 
document.querySelector("#fname"); const lname =
document.querySelector("#lname"); const dob =
document.querySelector("#dob"); const select =
document.querySelector("select"); const country =
document.querySelector("#country"); const address =
document.querySelector("#address"); const phone =
document.querySelector("#phone");
const password = document.querySelector("#password");
const register = document.querySelector("#register");
const imale = document.querySelector("#Savings"); 
const ifemale = document.querySelector("#Current"); 
const iother = document.querySelector("Loan");
const fname_pat = new RegExp("[A-Z][a-z]*"); const
lname_pat = new RegExp("[A-Z][a-z]*"); const
country_pat = new RegExp("[A-Z][a-z]*"); const
city_pat = new RegExp("[A-Z][a-z]*");
const dob_pat = new RegExp("[0-3][0-9]-[0-1][0-9]-[1-2][90][0-9][0-9]");
function validateForm() { if
(!fname_pat.test(fname.value)) {
alert("First name should be entered with First letter capital");
return false;
}
if (!lname_pat.test(lname.value)) {
alert("Last name should be entered with First letter capital");
return false;
}
if (!dob_pat.test(dob.value)) {
alert("Enter DOB in correct pattern");
return false;
}
if (!country_pat.test(country.value)) {
alert("Country name should be entered with First letter capital");
return false;
}
if (!city_pat.test(city.value)) {
alert("City name should be entered with First letter capital");
return false;
}
if (!(Savings.checked || Current.checked || Loan.checked)) {
alert("Select atleast one type of account"); return false;
}
if (parseInt(phone.value.length) != 10) {
alert("Phone digit number must be only 10 digits");
return false;
}
}
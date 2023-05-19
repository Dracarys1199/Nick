<h1>Total Amount of Selected Products</h1>

<%@ page import="java.util.*" %>
<%
// get the selected products from the request parameter
String[] selectedProducts = request.getParameterValues("selectedProducts");

// initialize the total price to 0
double totalPrice = 0.0;

// loop through the selected products and add their prices to the total
if (selectedProducts != null) {
    %>
    <TABLE BORDER="1">
        <TR>
            <TH>Product Name</TH>
            <TH>Product Price</TH>
        </TR>
    <% 
    for (String product : selectedProducts) {
        String[] parts = product.split(":");
        double price = Double.parseDouble(parts[1]);
        totalPrice += price;
    %>
        <TR>
            <TD> <%= parts[0] %></TD>
            <TD> <%= price %></td>
        </TR>
    <% }
    %>
    </TABLE>
<% } %>



<!DOCTYPE html>
<html>
<head>
    <title>Total Amount of Selected Products</title>
</head>
<body>
    <p> total amount of the selected products is <b><%= totalPrice %></b>.</p>
</body>
</html>
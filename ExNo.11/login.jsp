<%@ page import="java.sql.*" %>
<%@ page import="java.io.*" %> 
<%  Class.forName("com.mysql.jdbc.Driver");%>

<%
   // Create cookies for first and last names.      
   Cookie email = new Cookie("email", request.getParameter("email"));
   
   // Set expiry date after 24 Hrs for both the cookies.
   email.setMaxAge(60*60*24); 
   
   // Add both the cookies in the response header.
   response.addCookie( email );
%>

<html>
   <head>
    <title>Cart</title>
   </head>
   
   <body>
        <header style = 'text-align: right'><%= request.getParameter("email")%>
            &nbsp;&nbsp;&nbsp;&nbsp;
        <a href="logout.jsp" target="_blank">Logout</a> 
        </header><hr style = "border:2px solid black">

        <%
            Connection connection = DriverManager.getConnection("jdbc:mysql://localhost:3306/cart", "root", "");
            Statement statement = connection.createStatement() ;
            ResultSet resultset = statement.executeQuery("select * from products") ;
        %>
        <form method="post" action="Total.jsp">
        <TABLE BORDER="1">
            <TR>
            <TH>Product Name</TH>
            <TH>Product Price</TH>
            <TH>Checked</TH>
            </TR>
            <% while(resultset.next()){ %>
            <TR>
             <TD> <%= resultset.getString("name") %></td>
             <TD> <%= resultset.getString("price") %></TD>
             <td><input type="checkbox" name="selectedProducts" value='<%= resultset.getString("name") + ":" + resultset.getString("price") %>'></td>
            </TR>
            <% } %>
        </TABLE>
        <br><br>
        <input type="submit" value="Calculate Total">
    </form>
   </body>
</html>
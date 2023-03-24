import java.io.*; 
import javax.servlet.*;
import javax.servlet.http.*;
// Extend HttpServlet class
public class ex5b extends HttpServlet {
// Method to handle GET method request. public void
doGet(HttpServletRequest request, HttpServletResponse response) throws
ServletException, IOException {
// Set response content type
response.setContentType("text/html");
//To use html in java
PrintWriter out = response.getWriter();
String title = "Review";
String docType =
"<!doctype html public \"-//w3c//dtd html 4.0 " +
"transitional//en\">\n";
//HTML in Java
out.println(docType +
"<html>\n" +
//Title
"<head><title>" + title + "</title></head>\n" +
//Content

"<body>\n"+
//Header
"<h1 align=\"center\">" + title + "</h1>\n" +
//Displaying entered details from form
"<ul>\n" +
"<li>Name: " + request.getParameter("name") + "</li>\n" +
"<li>Username: " + request.getParameter("username") + "</li>\n" +
"<li>Date of Birth: " + request.getParameter("dob") + "</li>\n" +
"<li>Country: " + request.getParameter("country") + "</li>\n" +
"<li>City: " + request.getParameter("city") + "</li>\n" +
"<li>Email: " + request.getParameter("email") + "</li>\n" +
"<li>Password: " + request.getParameter("pwd") + "</li>\n" +
"<li>Phone number: " + request.getParameter("phone") + "</li>\n" +
"</ul>\n" +
"</body>" +
"</html>");
};
public void doPost(HttpServletRequest request, HttpServletResponse response)
throws ServletException, IOException {
doGet(request, response);
}
}
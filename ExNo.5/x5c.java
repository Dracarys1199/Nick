//Importing required packages
import java.io.*; 
import javax.servlet.*;
import javax.servlet.http.*;
// Extend HttpServlet class
public class ex5c extends HttpServlet {
// Method to handle GET method request. public void
doGet(HttpServletRequest request, HttpServletResponse response) throws
ServletException, IOException {
// Set response content type
response.setContentType("text/html");
//To use HTML in Java
PrintWriter out = response.getWriter();
String title = "Converter";
String docType =
"<!doctype html public \"-//w3c//dtd html 4.0 " +
"transitional//en\">\n";
//Converting Rupee from string to float and then converting to dollar
Float dollar = Float.parseFloat(request.getParameter("rupee")) / 75;
//HTML in Java
out.println(docType +
"<html>\n" +
//Title
"<head><title>" + title + "</title></head>\n" +
"<body>\n" +
//Header
"<h1 align=\"center\">Rupee to Dollar Converter</h1>\n" +
//Getting rupee from ip address
"<p align=\"center\">Entered amount in Rupees: " + request.getParameter("
rupee") + "</p>\n" +
//Displaying converted dollar
"<p align=\"center\">Rupees to Dollars: " + dollar + "</p>\n" +
"</body>" +
"</html>");
}
public void doPost(HttpServletRequest request, HttpServletResponse response)
throws ServletException, IOException {
doGet(request, response);
}
}
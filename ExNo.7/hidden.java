import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;


// Extend HttpServlet class
public class hidden extends HttpServlet {


   // Method to handle GET method request.
   public void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {
     
      // Set response content type
        response.setContentType("text/html");


        PrintWriter out = response.getWriter();


        String name = request.getParameter("name");
           
        out.println("Welcome " + name);


        out.println("<form action='hidden2' method = 'POST' >");
        out.println("<input type = 'hidden'  name = 'name' value='"+ name +"'>");
        out.println("<input type = 'submit' value = 'Submit'  />");
        out.println("</form>");
   }
}
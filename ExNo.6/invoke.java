import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;


// Extend HttpServlet class
public class invoke extends HttpServlet {


   // Method to handle GET method request.
   public void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {
     
      // Set response content type
      response.setContentType("text/html");


      PrintWriter out = response.getWriter();


      String name = request.getParameter("name");
      String age = request.getParameter("age");
         
      out.println("Name : " + name + "<br> Age :" + age);
   }
}
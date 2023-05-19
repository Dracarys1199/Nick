import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;


// Extend HttpServlet class
public class hidden2 extends HttpServlet {


   // Method to handle GET method request.
   public void doPost(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {
     
      // Set response content type
        response.setContentType("text/html");


        PrintWriter out = response.getWriter();


        String name1 = request.getParameter("name");
           
        out.println("Welcome " + name1);
   }
}
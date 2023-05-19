import java.io.*;
import java.sql.Date;
import java.util.*;
import javax.servlet.*;
import javax.servlet.http.*;


public class hitCount extends HttpServlet {


   private int hitCount;


   public void init() {
      // Reset hit counter.
      hitCount = 0;
   }


   public void doGet(HttpServletRequest request, HttpServletResponse response)
      throws ServletException, IOException {


      // Set response content type
      response.setContentType("text/html");


      // This method executes whenever the servlet is hit
      // increment hitCount
      hitCount++;
      PrintWriter out = response.getWriter();
     
      out.println( "Hit Count = " + hitCount );
   }
   
}
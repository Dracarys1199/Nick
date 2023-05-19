import java.io.*;
import java.util.*;
import javax.servlet.*;
import javax.servlet.http.*;
import java.sql.*;


public class update extends HttpServlet  {
   
   
    Connection con;
    PreparedStatement pst;
    ResultSet rs;
    int row;


    public void doPost(HttpServletRequest req,HttpServletResponse rsp ) throws IOException,ServletException
    {
       
        rsp.setContentType("text/html");
        PrintWriter out = rsp.getWriter();


        try {
            Class.forName("com.mysql.jdbc.Driver");
            con = DriverManager.getConnection("jdbc:mysql://localhost/aai","root","");
           
            String email = req.getParameter("email");


            out.println("<form method='post' action='update1' target='_blank'>");
            out.println("<input type = 'hidden'  name = 'email' value='"+ email +"'>");
            out.println("Name:<input type='text' id='name' name='name'><br>");
            out.println("Contact:<input type='text' id='contact' name='contact'><br>");
            out.println("Address:<input type='text' id='address' name='address'><br>");
            out.println("<input type='submit' value='Submit'></input>");
            out.println("</form>");


        }  catch(Exception e)
        {
            System.out.print("Do not connect to DB - Error:"+e);
        }


    }
 
}
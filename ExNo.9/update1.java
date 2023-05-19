import java.io.*;
import java.util.*;
import javax.servlet.*;
import javax.servlet.http.*;
import java.sql.*;


public class update1 extends HttpServlet  {
   
   
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
            String name = req.getParameter("name");
            String contact = req.getParameter("contact");
            String address = req.getParameter("address");
            pst = con.prepareStatement("update details set name = ?, contact = ?, address = ? where email = ?");
            pst.setString(1, name);
            pst.setString(2, contact);
            pst.setString(3, address);
            pst.setString(4, email);
            row = pst.executeUpdate();
           
            out.println("<h1>Record Updateeeedd   </h1>");


        }  catch(Exception e)
        {
            System.out.print("Do not connect to DB - Error:"+e);
        }


    }
 
}
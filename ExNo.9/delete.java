import java.io.*;
import java.sql.*;
import java.util.*;
import javax.servlet.*;
import javax.servlet.http.*;


public class delete extends HttpServlet {
   
    Connection con;
    PreparedStatement pst;
    ResultSet rs;
    int row;
   
   
   
    public void doGet(HttpServletRequest req,HttpServletResponse rsp ) throws IOException,ServletException
    {
       
        rsp.setContentType("text/html");
        PrintWriter out = rsp.getWriter();
       
        String email = req.getParameter("email");
               
        try {
             Class.forName("com.mysql.jdbc.Driver");
             con = DriverManager.getConnection("jdbc:mysql://localhost/aai","root","");
             pst = con.prepareStatement("delete from details where email = ?");
             pst.setString(1, email);
             row = pst.executeUpdate();
             
             out.println("<h1>Record inserted successfully</h1>");


           
        } catch (ClassNotFoundException e) {
            out.println("<h1>Error: MySQL JDBC driver not found</h1>");
          } catch (SQLException e) {
            out.println("<h1>Error: " + e.getMessage() + "</h1>");
          } finally {
            // Clean up database resources
            try {
              if (pst != null) {
                pst.close();
              }
              if (con != null) {
                con.close();
              }
            } catch (SQLException e) {
              out.println("<h1>Error closing database resources: " + e.getMessage() + "</h1>");
            }
          }
   
   
    }
   
}
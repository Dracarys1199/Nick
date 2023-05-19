import java.io.*;
import java.sql.*;
import javax.servlet.*;
import javax.servlet.http.*;


public class data extends HttpServlet {
 
  public void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    response.setContentType("text/html");
    PrintWriter out = response.getWriter();
   
    // Retrieve the form data from the request
   
    String name = request.getParameter("name");
    String email = request.getParameter("email");
    String password = request.getParameter("pass");
    String contact = request.getParameter("contact");
    String address = request.getParameter("address");
   
    // Insert the data into the database
    Connection conn = null;
    PreparedStatement stmt = null;
    try {
      // Load the MySQL JDBC driver
      Class.forName("com.mysql.jdbc.Driver");
     
      // Create a connection to the database
      conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/aai", "root", "");
     
      // Prepare the SQL statement for insertion
      String sql = "INSERT INTO details (name, email, password, contact, address) VALUES (?, ?, ?, ?, ?)";
      stmt = conn.prepareStatement(sql);
      stmt.setString(1, name);
      stmt.setString(2, email);
      stmt.setString(3, password);
      stmt.setString(4, contact);
      stmt.setString(5, address);
     
      // Execute the SQL statement
      int rowsAffected = stmt.executeUpdate();
     
      // Check if the insertion was successful
      if (rowsAffected > 0) {
        out.println("<h1>Record inserted successfully</h1>");
      } else {
        out.println("<h1>Error inserting record</h1>");
      }
    } catch (ClassNotFoundException e) {
      out.println("<h1>Error: MySQL JDBC driver not found</h1>");
    } catch (SQLException e) {
      out.println("<h1>Error: " + e.getMessage() + "</h1>");
    } finally {
      // Clean up database resources
      try {
        if (stmt != null) {
          stmt.close();
        }
        if (conn != null) {
          conn.close();
        }
      } catch (SQLException e) {
        out.println("<h1>Error closing database resources: " + e.getMessage() + "</h1>");
      }
    }
  }
}
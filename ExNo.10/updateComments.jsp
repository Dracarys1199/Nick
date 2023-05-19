<%@ page import="java.sql.*" %>

<%
    // Retrieve the submitted form data
    String title = request.getParameter("title");
    String comments = request.getParameter("comments");

    try {
        // Establish a database connection
        Class.forName("com.mysql.jdbc.Driver");
        String url = "jdbc:mysql://localhost:3306/aai";
        String username = "root";
        String password = "";
        Connection conn = DriverManager.getConnection(url, username, password);

        // Create and execute the SQL query to update the comments
        String query = "UPDATE movies SET comments = ? WHERE title = ?";
        PreparedStatement stmt = conn.prepareStatement(query);
        stmt.setString(1, comments);
        stmt.setString(2, title);
        int rowsAffected = stmt.executeUpdate();

        // Close the database resources
        stmt.close();
        conn.close();

        // Redirect back to the movie details page
        response.sendRedirect("movies.jsp");
    } catch (Exception e) {
        out.println("An error occurred: " + e.getMessage());
    }
%>
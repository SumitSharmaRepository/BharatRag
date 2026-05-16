export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: "#534AB7",
          light:   "#EEEDFE",
          dark:    "#3C3489",
        }
      }
    },
  },
  plugins: [],
}
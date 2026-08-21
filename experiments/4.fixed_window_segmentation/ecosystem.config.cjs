// pm2 config cho experiment 4. Chạy: pm2 start ecosystem.config.cjs
// Giới hạn số file mỗi dataset: LIMIT=5 pm2 start ecosystem.config.cjs
//
// interpreter "none" + launcher src/run.sh: pm2 6.x chọn wrapper bằng
// interpreter.includes("bun"), nên mọi python dưới /home/ubuntu bị nhận
// nhầm là Bun và crash. Để bash tự lo việc chọn python của env.
const path = require("path");

module.exports = {
  apps: [
    {
      name: "exp4-fixed-window",
      script: path.join(__dirname, "src", "run.sh"),
      cwd: path.join(__dirname, "src"),
      interpreter: "none",
      args: process.env.LIMIT ? ["--limit", process.env.LIMIT] : [],
      autorestart: false,
    },
  ],
};

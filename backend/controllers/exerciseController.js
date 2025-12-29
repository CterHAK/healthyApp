const exerciseService = require("../services/exerciseService");

// ----------------- POST: Tạo group + thêm chi tiết -----------------
async function addExerciseGroupAndDetail(req, res) {
  try {
    const { email, group_name, exercises } = req.body;

    if (!email || !group_name || !exercises || exercises.length === 0) {
      return res.status(400).json({
        status: "error",
        message: "Thiếu email, group_name hoặc exercises"
      });
    }

    // Tạo group nếu chưa tồn tại
    let group = await exerciseService.getGroup(email, group_name);
    if (!group) {
      group = await exerciseService.addExerciseGroup({ email, group_name });
    }

    // Thêm hoặc cập nhật chi tiết bài tập
    const savedDetails = [];
    for (const ex of exercises) {
      const { exercise_name, sets = 3, reps = 10 } = ex;
      let detail = await exerciseService.getExerciseDetail(email, group_name, exercise_name);

      if (!detail) {
        detail = await exerciseService.addExerciseDetail({ email, group_name, name: exercise_name, sets, reps });
      } else {
        detail.sets = sets;
        detail.reps = reps;
        await detail.save();
      }
      savedDetails.push(detail);
    }

    res.json({
      status: "success",
      message: "Tạo nhóm bài tập thành công",
      group,
      details: savedDetails
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ status: "error", message: e.toString() });
  }
}

// ----------------- GET: Lấy group + chi tiết -----------------
async function getExerciseGroup(req, res) {
  try {
    const { email } = req.query;
    if (!email) return res.status(400).json({ status: "error", message: "Thiếu email" });

    const groups = await exerciseService.getGroupsByEmail(email) || [];
    const result = [];

    for (const g of groups) {
      const details = await exerciseService.getExercisesByEmailAndGroup(email, g.group_name);
      result.push({ group: g, details });
    }

    res.json({ status: "success", groups: result });
  } catch (e) {
    console.error(e);
    res.status(500).json({ status: "error", message: e.toString() });
  }
}

// ----------------- POST: Thêm vào Plan -----------------
async function addExercisePlan(req, res) {
  try {
    const { email, group_name, day, session } = req.body;
    if (!email || !group_name || !day || !session) {
      return res.status(400).json({
        status: "error",
        message: "Thiếu email, group_name, day hoặc session"
      });
    }

    const plan = await exerciseService.addExercisePlan({
      email,
      group_name,
      day,
      session,
      done_flag: false
    });

    res.json({
      status: "success",
      message: "Thêm vào plan thành công",
      plan
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ status: "error", message: e.toString() });
  }
}

// ----------------- GET: Lấy Plan theo email hoặc ngày -----------------
async function getExercisePlans(req, res) {
  try {
    const { email, day } = req.query;
    if (!email) 
      return res.status(400).json({ status: "error", message: "Thiếu email" });

    // Lấy plan theo email hoặc email + day
    let plans;
    if (day) {
      plans = await exerciseService.getPlansByEmailAndDay(email, day);
    } else {
      plans = await exerciseService.getPlansByEmail(email);
    }

    // Lấy chi tiết bài tập cho từng plan
    const plansWithDetails = await Promise.all(
      plans.map(async plan => {
        const details = await exerciseService.getExercisesByEmailAndGroup(email, plan.group_name);
        return { ...plan, details };
      })
    );

    res.json({ status: "success", plans: plansWithDetails });
  } catch (e) {
    console.error(e);
    res.status(500).json({ status: "error", message: e.toString() });
  }
}


// ----------------- Các chức năng khác (giữ nguyên) -----------------
async function filterExercises(req, res) {
  try {
    const { muscle, equipment, difficulty } = req.query;
    const data = await exerciseService.filterExercises({
      muscle,
      equipment,
      level: difficulty ? Number(difficulty) : undefined
    });
    res.json(data);
  } catch (e) {
    res.status(500).json({ status: "error", message: e.toString() });
  }
}

async function getDependentFilters(req, res) {
  try {
    const { muscle, equipment, difficulty } = req.query;
    const data = await exerciseService.getDependentFilters({
      muscle,
      equipment,
      level: difficulty ? Number(difficulty) : undefined
    });
    res.json({ status: "success", ...data });
  } catch (e) {
    res.status(500).json({ status: "error", message: e.toString() });
  }
}

async function getMuscles(req, res) {
  try {
    const data = await exerciseService.getMuscles();
    res.json({ status: "success", muscle_groups: data.muscle_groups || [] });
  } catch (e) {
    res.status(500).json({ status: "error", message: e.toString() });
  }
}

async function getEquipment(req, res) {
  try {
    const data = await exerciseService.getEquipment();
    res.json({ status: "success", equipment_list: data.equipment_list || [] });
  } catch (e) {
    res.status(500).json({ status: "error", message: e.toString() });
  }
}

async function getExerciseInfo(req, res) {
  try {
    const { exercise_name } = req.query;
    if (!exercise_name) return res.status(400).json({ status: "error", message: "Thiếu tên bài tập" });

    const data = await exerciseService.getExerciseInfoByName(exercise_name);
    res.json(data);
  } catch (e) {
    res.status(500).json({ status: "error", message: e.toString() });
  }
}

async function calculateCalories(req, res) {
  try {
    const { exercise_name, sets, reps, weight_kg } = req.body;
    if (!exercise_name || !sets || !reps) {
      return res.status(400).json({ status: "error", message: "Thiếu exercise_name, sets, reps" });
    }

    const result = await exerciseService.estimateCaloriesByName({
      exercise_name,
      sets: Number(sets),
      reps: Number(reps),
      weight_kg: Number(weight_kg || 70)
    });

    res.json(result);
  } catch (e) {
    res.status(500).json({ status: "error", message: e.toString() });
  }
}
// ✅ Cập nhật done_flag cho một plan
async function markWorkoutDone(req, res) {
  try {
    const { email, group_name, day, session, weight = 70 } = req.body;

    if (!email || !group_name || !day || !session) {
      return res.status(400).json({ success: false, message: "Thiếu dữ liệu gửi lên" });
    }

    // ⭐ Pass weight để tính calories
    const updated = await exerciseService.updateDoneFlag(
      email, 
      group_name, 
      day, 
      session,
      parseFloat(weight)
    );

    if (!updated) {
      return res.status(404).json({ success: false, message: "Không tìm thấy kế hoạch để cập nhật" });
    }

    return res.status(200).json({
      success: true,
      message: "Cập nhật done_flag thành công",
      data: updated
    });

  } catch (error) {
    console.error("Error markWorkoutDone:", error);
    res.status(500).json({
      success: false,
      message: "Lỗi cập nhật done_flag",
      details: error.message
    });
  }
}

// ----------------- EXPORT -----------------
module.exports = {
  addExerciseGroupAndDetail,
  getExerciseGroup,
  addExercisePlan,
  getExercisePlans,
  filterExercises,
  getDependentFilters,
  getMuscles,
  getEquipment,
  getExerciseInfo,
  calculateCalories,
  markWorkoutDone
};

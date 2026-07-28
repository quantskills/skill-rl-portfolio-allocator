#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${PROJECT_DIR}/rl-portfolio-allocator"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_credentials() {
    if [[ -z "$PANDA_DATA_USERNAME" || -z "$PANDA_DATA_PASSWORD" ]]; then
        log_error "PANDA_DATA_USERNAME 或 PANDA_DATA_PASSWORD 未设置"
        echo "请运行以下命令设置凭据:"
        echo "  export PANDA_DATA_USERNAME=<your-username>"
        echo "  export PANDA_DATA_PASSWORD=<your-password>"
        exit 1
    fi
    log_info "Panda Data 凭据已设置"
}

check_dependencies() {
    log_info "检查依赖..."

    if ! python -c "import stable_baselines3" 2>/dev/null; then
        log_warn "stable_baselines3 未安装，正在安装..."
        pip install "stable-baselines3>=2.0" gymnasium
    fi

    if ! python -c "import gymnasium" 2>/dev/null; then
        log_error "gymnasium 安装失败"
        exit 1
    fi

    log_info "依赖检查完成"
}

run_features() {
    log_info "生成特征数据..."
    cd "$WORK_DIR"
    PYTHONPATH="$WORK_DIR:$PYTHONPATH" python -m scripts.features
    log_info "特征数据生成完成"
}

run_train() {
    log_info "训练模型..."
    cd "$WORK_DIR"
    PYTHONPATH="$WORK_DIR:$PYTHONPATH" python -m scripts.train --timesteps 200000
    log_info "模型训练完成"
}

run_backtest() {
    log_info "回测模型..."
    cd "$WORK_DIR"
    PYTHONPATH="$WORK_DIR:$PYTHONPATH" python -m scripts.backtest
    log_info "回测完成"
}

run_stress_test() {
    log_info "压力测试（4个市场阶段）..."
    cd "$WORK_DIR"
    PYTHONPATH="$WORK_DIR:$PYTHONPATH" python -m scripts.stress_test
    log_info "压力测试完成"
}

run_allocate() {
    log_info "生成配置（重新训练模式）..."
    cd "$WORK_DIR"
    PYTHONPATH="$WORK_DIR:$PYTHONPATH" python -m scripts.allocate --retrain
    log_info "配置生成完成"
}

run_validate() {
    log_info "验证模型..."
    cd "$WORK_DIR"
    PYTHONPATH="$WORK_DIR:$PYTHONPATH" python -m scripts.validate
    log_info "验证完成"
}

show_usage() {
    cat << EOF
用法: $0 [选项]

选项:
    --all              运行完整流程 (features -> train -> backtest -> stress_test -> allocate -> validate)
    --features         只生成特征数据
    --train            只训练模型
    --backtest         只运行回测
    --stress-test      只运行压力测试
    --allocate         只生成配置
    --validate         只验证模型
    --quick            快速流程 (features -> train -> backtest)
    --help             显示帮助信息

环境变量:
    PANDA_DATA_USERNAME  Panda Data 用户名（必需）
    PANDA_DATA_PASSWORD  Panda Data 密码（必需）
    TRAIN_DEVICE         训练设备 (cpu/cuda/mps, 默认自动检测)

示例:
    # 设置凭据并运行完整流程
    export PANDA_DATA_USERNAME=your_username
    export PANDA_DATA_PASSWORD=your_password
    $0 --all

    # 运行快速流程
    $0 --quick

    # 只生成特征
    $0 --features

EOF
}

main() {
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 0
    fi

    check_credentials
    check_dependencies

    case "$1" in
        --all)
            run_features
            run_train
            run_backtest
            run_stress_test
            run_allocate
            run_validate
            log_info "完整流程执行完成！"
            ;;
        --features)
            run_features
            ;;
        --train)
            run_train
            ;;
        --backtest)
            run_backtest
            ;;
        --stress-test)
            run_stress_test
            ;;
        --allocate)
            run_allocate
            ;;
        --validate)
            run_validate
            ;;
        --quick)
            run_features
            run_train
            run_backtest
            log_info "快速流程执行完成！"
            ;;
        --help)
            show_usage
            ;;
        *)
            log_error "未知选项: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"

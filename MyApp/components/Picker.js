import React from "react";

export function Picker({ children, ...props }) {
  return (
    <select {...props}>
      {React.Children.map(children, (child) => (
        <option value={child.props.value}>{child.props.label}</option>
      ))}
    </select>
  );
}

export function PickerItem({ label, value }) {
  return <option value={value}>{label}</option>;
}
